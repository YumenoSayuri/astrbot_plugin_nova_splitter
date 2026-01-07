"""
Nova Splitter - 智能消息分段插件
作者: Nova for 辉宝主人
功能: 
  1. 按字数均分分段
  2. 按标点分段
  3. LLM辅助智能分段
  4. 智能标点清理(边界清理/句内保留)
"""

import re
import math
import random
import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse, Provider
from astrbot.api.message_components import Plain, BaseMessageComponent, Reply


@dataclass
class SplitResult:
    """分段结果"""
    segments: List[str]
    split_points: List[int]  # 分割点位置


class SplitStrategy(ABC):
    """分段策略基类"""
    
    def __init__(self, config: dict):
        self.config = config
        # 配对符号映射
        self.pair_map = {
            '"': '"', '《': '》', '（': '）', '(': ')',
            '[': ']', '{': '}', "'": "'", '【': '】', '<': '>'
        }
        self.quote_chars = {'"', "'", '`'}
    
    @abstractmethod
    def split(self, text: str) -> SplitResult:
        """执行分段"""
        pass
    
    def is_inside_pair(self, text: str, pos: int) -> bool:
        """检查位置是否在配对符号内部"""
        stack = []
        for i, char in enumerate(text[:pos]):
            if char in self.quote_chars:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
            elif char in self.pair_map:
                stack.append(char)
            elif stack:
                expected = self.pair_map.get(stack[-1])
                if char == expected:
                    stack.pop()
        return len(stack) > 0
    
    def find_best_split_point(self, text: str, ideal_pos: int, search_range: int = 10) -> int:
        """在理想位置附近找到最佳分割点"""
        split_chars = self.config.get("split_punctuation", "。！？!?；;\n")
        
        # 搜索范围
        start = max(0, ideal_pos - search_range)
        end = min(len(text), ideal_pos + search_range)
        
        best_pos = ideal_pos
        best_score = -1
        
        for pos in range(start, end):
            if pos >= len(text):
                continue
            
            char = text[pos]
            score = 0
            
            # 优先级: 句末标点 > 逗号 > 其他位置
            if char in "。！？!?\n":
                score = 100
            elif char in "；;":
                score = 80
            elif char in "，,、":
                score = 60
            elif char in " ":
                score = 40
            
            # 位置越接近理想位置，加分越多
            distance_penalty = abs(pos - ideal_pos) * 2
            score -= distance_penalty
            
            # 如果在配对符号内，大幅减分
            if self.is_inside_pair(text, pos):
                score -= 200
            
            if score > best_score:
                best_score = score
                best_pos = pos
        
        return best_pos


class CharCountSplitter(SplitStrategy):
    """按字数均分策略"""
    
    def split(self, text: str) -> SplitResult:
        target_segments = self.config.get("target_segments", 3)
        min_segment_length = self.config.get("min_segment_length", 10)
        
        total_length = len(text)
        
        # 如果文本太短，不分段
        if total_length < min_segment_length * 2:
            return SplitResult(segments=[text], split_points=[])
        
        # 计算理想的每段长度
        ideal_length = total_length // target_segments
        
        # 如果理想长度太短，减少段数
        while ideal_length < min_segment_length and target_segments > 1:
            target_segments -= 1
            ideal_length = total_length // target_segments
        
        if target_segments == 1:
            return SplitResult(segments=[text], split_points=[])
        
        # 计算理想分割点
        split_points = []
        for i in range(1, target_segments):
            ideal_pos = i * ideal_length
            # 在理想位置附近找最佳分割点
            best_pos = self.find_best_split_point(text, ideal_pos)
            split_points.append(best_pos)
        
        # 去重并排序
        split_points = sorted(set(split_points))
        
        # 根据分割点切分文本
        segments = []
        prev_pos = 0
        for pos in split_points:
            if pos > prev_pos:
                segments.append(text[prev_pos:pos + 1])
                prev_pos = pos + 1
        
        # 添加最后一段
        if prev_pos < len(text):
            segments.append(text[prev_pos:])
        
        return SplitResult(segments=segments, split_points=split_points)


class PunctuationSplitter(SplitStrategy):
    """按标点分段策略（改进版）"""
    
    def split(self, text: str) -> SplitResult:
        split_chars = self.config.get("split_punctuation", "。！？!?；;\n")
        max_segments = self.config.get("max_segments", 7)
        enable_smart = self.config.get("enable_smart_split", True)
        
        # 编译正则
        pattern = f"[{re.escape(split_chars)}]+"
        
        if enable_smart:
            segments = self._smart_split(text, pattern)
        else:
            segments = self._simple_split(text, pattern)
        
        # 如果段数超过限制，合并后面的段
        if len(segments) > max_segments and max_segments > 0:
            final_segments = segments[:max_segments - 1]
            merged = "".join(segments[max_segments - 1:])
            final_segments.append(merged)
            segments = final_segments
        
        # 计算分割点（用于标点清理）
        split_points = []
        pos = 0
        for seg in segments[:-1]:
            pos += len(seg)
            split_points.append(pos - 1)  # 指向每段最后一个字符
        
        return SplitResult(segments=segments, split_points=split_points)
    
    def _simple_split(self, text: str, pattern: str) -> List[str]:
        """简单分割"""
        parts = re.split(f"({pattern})", text)
        segments = []
        current = ""
        
        for part in parts:
            if not part:
                continue
            if re.fullmatch(pattern, part):
                current += part
                if current.strip():
                    segments.append(current)
                current = ""
            else:
                current += part
        
        if current.strip():
            segments.append(current)
        
        return segments
    
    def _smart_split(self, text: str, pattern: str) -> List[str]:
        """智能分割（避免在配对符号内断句）"""
        compiled = re.compile(pattern)
        stack = []
        segments = []
        current = ""
        i = 0
        n = len(text)
        
        while i < n:
            char = text[i]
            
            # 处理引号
            if char in self.quote_chars:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    stack.append(char)
                current += char
                i += 1
                continue
            
            # 如果在配对符号内，不分割
            if stack:
                expected = self.pair_map.get(stack[-1])
                if char == expected:
                    stack.pop()
                elif char in self.pair_map:
                    stack.append(char)
                current += char
                i += 1
                continue
            
            # 配对符号开始
            if char in self.pair_map:
                stack.append(char)
                current += char
                i += 1
                continue
            
            # 检查是否匹配分割模式
            match = compiled.match(text, pos=i)
            if match:
                delimiter = match.group()
                current += delimiter
                if current.strip():
                    segments.append(current)
                current = ""
                i += len(delimiter)
            else:
                current += char
                i += 1
        
        if current.strip():
            segments.append(current)
        
        return segments


class LLMAssistSplitter(SplitStrategy):
    """LLM辅助分段策略"""
    
    def __init__(self, config: dict, context: Context):
        super().__init__(config)
        self.context = context
    
    async def split_async(self, text: str) -> SplitResult:
        """异步分段（需要调用LLM）"""
        target_segments = self.config.get("target_segments", 3)
        provider_id = self.config.get("llm_assist_provider_id", "")
        prompt_template = self.config.get("llm_split_prompt", 
            "请将以下文本分成{n}个段落，要求每段语义完整，长度尽量均匀。仅输出JSON数组格式，如: [\"段1\", \"段2\", \"段3\"]\n\n原文：\n{text}")
        
        # 获取Provider
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
        else:
            provider = self.context.get_using_provider()
        
        if not provider or not isinstance(provider, Provider):
            logger.warning("[Nova-Splitter] 未找到LLM Provider，回退到按字数分段")
            fallback = CharCountSplitter(self.config)
            return fallback.split(text)
        
        # 构建prompt
        prompt = prompt_template.format(n=target_segments, text=text)
        
        try:
            response = await provider.text_chat(
                prompt=prompt,
                session_id=uuid.uuid4().hex,
                persist=False
            )
            
            # 解析JSON响应
            result_text = response.completion_text.strip()
            # 尝试提取JSON数组
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                import json
                segments = json.loads(json_match.group())
                if isinstance(segments, list) and all(isinstance(s, str) for s in segments):
                    return SplitResult(segments=segments, split_points=[])
            
            logger.warning(f"[Nova-Splitter] LLM返回格式异常，回退到按字数分段: {result_text[:100]}")
        except Exception as e:
            logger.error(f"[Nova-Splitter] LLM分段失败: {e}")
        
        # 回退到按字数分段
        fallback = CharCountSplitter(self.config)
        return fallback.split(text)
    
    def split(self, text: str) -> SplitResult:
        """同步接口（回退到按字数分段）"""
        fallback = CharCountSplitter(self.config)
        return fallback.split(text)


class PunctuationProcessor:
    """标点处理引擎"""
    
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("punctuation_clean_mode", "edge_only")
        self.edge_chars = config.get("edge_clean_chars", "，,、")
        self.space_chars = config.get("space_replace_chars", "")
        self.custom_regex = config.get("custom_clean_regex", "")
    
    def process(self, segments: List[str], split_points: List[int]) -> List[str]:
        """处理分段后的标点"""
        if self.mode == "preserve":
            return segments
        
        processed = []
        for i, seg in enumerate(segments):
            if self.mode == "edge_only":
                seg = self._clean_edge(seg, is_last=(i == len(segments) - 1))
            elif self.mode == "replace_space":
                seg = self._replace_with_space(seg)
            elif self.mode == "custom":
                seg = self._custom_clean(seg)
            processed.append(seg)
        
        return processed
    
    def _clean_edge(self, text: str, is_last: bool = False) -> str:
        """仅清理边界标点"""
        if not text:
            return text
        
        # 清理开头的指定标点
        while text and text[0] in self.edge_chars:
            text = text[1:]
        
        # 清理结尾的指定标点（最后一段不清理句末标点）
        if not is_last:
            while text and text[-1] in self.edge_chars:
                text = text[:-1]
        
        return text
    
    def _replace_with_space(self, text: str) -> str:
        """将指定标点替换为空格"""
        if not self.space_chars:
            return text
        pattern = f"[{re.escape(self.space_chars)}]"
        return re.sub(pattern, " ", text)
    
    def _custom_clean(self, text: str) -> str:
        """自定义正则清理"""
        if not self.custom_regex:
            return text
        try:
            return re.sub(self.custom_regex, "", text)
        except re.error as e:
            logger.error(f"[Nova-Splitter] 自定义正则错误: {e}")
            return text


@register("nova-splitter", "Nova", "智能消息分段插件 - 支持按字数/标点/LLM分段，智能标点清理", "1.0.0")
class NovaSplitterPlugin(Star):
    """Nova智能分段插件"""
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.punctuation_processor = PunctuationProcessor(dict(config))
        logger.info("[Nova-Splitter] 插件已初始化")
        logger.info(f"[Nova-Splitter] 分段模式: {config.get('split_mode', 'char_count')}")
        logger.info(f"[Nova-Splitter] 标点清理模式: {config.get('punctuation_clean_mode', 'edge_only')}")
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """标记LLM响应"""
        setattr(event, "__is_llm_reply", True)
    
    @filter.on_decorating_result(priority=-100000000000000000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """处理消息分段"""
        # 防重入检查
        if getattr(event, "__nova_splitter_processed", False):
            return
        
        result = event.get_result()
        if not result or not result.chain:
            return
        
        # 作用范围检查
        split_scope = self.config.get("split_scope", "llm_only")
        is_llm_reply = getattr(event, "__is_llm_reply", False)
        
        if split_scope == "llm_only" and not is_llm_reply:
            return
        
        # 提取文本内容
        full_text = ""
        for comp in result.chain:
            if isinstance(comp, Plain):
                full_text += comp.text
        
        if not full_text.strip():
            return
        
        # 长度检查
        min_length = self.config.get("max_length_no_split", 50)
        if len(full_text) < min_length:
            return
        
        # 标记已处理
        setattr(event, "__nova_splitter_processed", True)
        
        # 选择分段策略
        split_mode = self.config.get("split_mode", "char_count")
        
        if split_mode == "llm_assist" and self.config.get("enable_llm_assist", False):
            splitter = LLMAssistSplitter(dict(self.config), self.context)
            split_result = await splitter.split_async(full_text)
        elif split_mode == "punctuation":
            splitter = PunctuationSplitter(dict(self.config))
            split_result = splitter.split(full_text)
        elif split_mode == "hybrid":
            # 混合模式：先按标点，如果段数不够再按字数
            punct_splitter = PunctuationSplitter(dict(self.config))
            split_result = punct_splitter.split(full_text)
            target = self.config.get("target_segments", 3)
            if len(split_result.segments) < target:
                char_splitter = CharCountSplitter(dict(self.config))
                split_result = char_splitter.split(full_text)
        else:  # char_count
            splitter = CharCountSplitter(dict(self.config))
            split_result = splitter.split(full_text)
        
        segments = split_result.segments
        
        # 如果只有一段，不处理
        if len(segments) <= 1:
            return
        
        # 标点处理
        segments = self.punctuation_processor.process(segments, split_result.split_points)
        
        # 过滤空段
        segments = [s for s in segments if s.strip()]
        
        if len(segments) <= 1:
            return
        
        logger.info(f"[Nova-Splitter] 消息被分为 {len(segments)} 段")
        
        # 处理非文本组件
        other_components = [c for c in result.chain if not isinstance(c, Plain)]
        
        # 构建分段消息链
        message_segments = []
        for i, seg_text in enumerate(segments):
            seg_chain = [Plain(seg_text)]
            # 第一段添加其他组件（如回复引用）
            if i == 0:
                for comp in other_components:
                    if isinstance(comp, Reply):
                        seg_chain.insert(0, comp)
                    else:
                        seg_chain.append(comp)
            message_segments.append(seg_chain)
        
        # 添加回复引用到第一段（如果没有）
        enable_reply = self.config.get("enable_reply", True)
        if enable_reply and message_segments and event.message_obj.message_id:
            has_reply = any(isinstance(c, Reply) for c in message_segments[0])
            if not has_reply:
                message_segments[0].insert(0, Reply(id=event.message_obj.message_id))
        
        # 发送前 N-1 段
        for i in range(len(message_segments) - 1):
            seg_chain = message_segments[i]
            seg_text = "".join([c.text for c in seg_chain if isinstance(c, Plain)])
            
            if not seg_text.strip():
                continue
            
            try:
                self._log_segment(i + 1, len(message_segments), seg_chain)
                
                mc = MessageChain()
                mc.chain = seg_chain
                await self.context.send_message(event.unified_msg_origin, mc)
                
                # 延迟
                delay = self._calculate_delay(seg_text)
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"[Nova-Splitter] 发送分段 {i+1} 失败: {e}")
        
        # 最后一段交给框架
        last_chain = message_segments[-1]
        self._log_segment(len(message_segments), len(message_segments), last_chain, "交给框架")
        
        result.chain.clear()
        result.chain.extend(last_chain)
    
    def _log_segment(self, index: int, total: int, chain: List[BaseMessageComponent], method: str = "主动发送"):
        """日志输出"""
        content = ""
        for comp in chain:
            if isinstance(comp, Plain):
                content += comp.text
            else:
                content += f"[{type(comp).__name__}]"
        
        content = content.replace('\n', '\\n')[:100]
        logger.info(f"[Nova-Splitter] 第 {index}/{total} 段 ({method}): {content}...")
    
    def _calculate_delay(self, text: str) -> float:
        """计算延迟时间"""
        strategy = self.config.get("delay_strategy", "linear")
        
        if strategy == "random":
            return random.uniform(
                self.config.get("random_min", 0.5),
                self.config.get("random_max", 2.0)
            )
        elif strategy == "log":
            base = self.config.get("log_base", 0.3)
            factor = self.config.get("log_factor", 0.5)
            return min(base + factor * math.log(len(text) + 1), 3.0)
        elif strategy == "linear":
            base = self.config.get("linear_base", 0.3)
            factor = self.config.get("linear_factor", 0.05)
            return min(base + len(text) * factor, 3.0)
        else:  # fixed
            return self.config.get("fixed_delay", 1.0)