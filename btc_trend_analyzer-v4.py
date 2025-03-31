import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import argparse
import os
import json
from tqdm import tqdm
import talib
import logging
import concurrent.futures
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class DataManager:
    """
    数据管理器类
    
    负责获取、缓存和预处理市场数据
    """
    
    def __init__(self, exchange_id: str, symbol: str, cache_dir: str, logger: logging.Logger):
        """
        初始化数据管理器
        
        参数:
            exchange_id: 交易所ID
            symbol: 交易对
            cache_dir: 缓存目录路径
            logger: 日志记录器
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.cache_dir = cache_dir
        self.logger = logger
        
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            self.logger.info(f"创建缓存目录: {cache_dir}")
            
        # 初始化交易所连接
        self.exchange = None
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            self.logger.info(f"已连接到交易所: {exchange_id}")
        except Exception as e:
            self.logger.error(f"连接交易所失败: {e}")
            raise
            
    def get_cache_file_path(self, timeframe: str) -> str:
        """
        获取指定时间周期的缓存文件路径
        
        参数:
            timeframe: 时间周期
            
        返回:
            str: 缓存文件路径
        """
        cache_file = os.path.join(
            self.cache_dir, 
            f"{self.exchange_id}_{self.symbol.replace('/', '_')}_{timeframe}.csv"
        )
        return cache_file.replace(':', '')
    
    def get_metadata_file_path(self, timeframe: str) -> str:
        """
        获取指定时间周期的元数据文件路径
        
        参数:
            timeframe: 时间周期
            
        返回:
            str: 元数据文件路径
        """
        metadata_file = os.path.join(
            self.cache_dir, 
            f"{self.exchange_id}_{self.symbol.replace('/', '_')}_{timeframe}_metadata.json"
        )
        return metadata_file.replace(':', '')
        
    def load_cached_data(self, timeframe: str) -> Tuple[Optional[pd.DataFrame], dict]:
        """
        加载缓存的数据和元数据
        
        参数:
            timeframe: 时间周期
            
        返回:
            Tuple[Optional[pd.DataFrame], dict]: DataFrame和元数据字典
        """
        cache_file = self.get_cache_file_path(timeframe)
        metadata_file = self.get_metadata_file_path(timeframe)
        
        # 默认元数据
        metadata = {
            'last_updated': None,
            'last_timestamp': None,
            'first_timestamp': None,
            'count': 0
        }
        
        # 尝试加载元数据
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                # 确保元数据中的时间字段是字符串类型
                for key in ['last_updated', 'last_timestamp', 'first_timestamp']:
                    if key in metadata and metadata[key] is not None and not isinstance(metadata[key], str):
                        metadata[key] = str(metadata[key])
                        
            except Exception as e:
                self.logger.warning(f"无法加载元数据文件: {e}")
        
        # 尝试加载数据
        df = None
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 验证数据完整性
                if df.empty:
                    self.logger.warning(f"缓存文件为空: {cache_file}")
                    df = None
                else:
                    # 确保排序
                    df = df.sort_values('timestamp')
                    
                    # 更新元数据以防与实际数据不符
                    if 'timestamp' in df.columns:
                        metadata['first_timestamp'] = df['timestamp'].min().isoformat()
                        metadata['last_timestamp'] = df['timestamp'].max().isoformat()
                        metadata['count'] = len(df)
                        # 确保last_updated也是字符串格式
                        if 'last_updated' not in metadata or metadata['last_updated'] is None:
                            metadata['last_updated'] = datetime.now().isoformat()
                
            except Exception as e:
                self.logger.error(f"加载缓存数据出错: {e}")
                df = None
                
                # 尝试备份损坏的文件
                try:
                    backup_file = cache_file + f".bak.{int(time.time())}"
                    os.rename(cache_file, backup_file)
                    self.logger.info(f"已将损坏的缓存文件备份为: {backup_file}")
                except Exception:
                    pass
                
        return df, metadata
    
    def save_data_and_metadata(self, timeframe: str, df: pd.DataFrame, metadata: dict) -> bool:
        """
        保存数据和元数据到缓存
        
        参数:
            timeframe: 时间周期
            df: 要保存的DataFrame
            metadata: 元数据字典
            
        返回:
            bool: 保存是否成功
        """
        if df is None or df.empty:
            self.logger.warning("无数据可保存")
            return False
            
        cache_file = self.get_cache_file_path(timeframe)
        metadata_file = self.get_metadata_file_path(timeframe)
        
        try:
            # 保存数据
            df.to_csv(cache_file, index=False)
            
            # 更新并保存元数据
            metadata['last_updated'] = datetime.now().isoformat()
            metadata['count'] = len(df)
            
            if 'timestamp' in df.columns:
                metadata['first_timestamp'] = df['timestamp'].min().isoformat()
                metadata['last_timestamp'] = df['timestamp'].max().isoformat()
                
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"已保存{timeframe}数据和元数据")
            return True
            
        except Exception as e:
            self.logger.error(f"保存数据和元数据出错: {e}")
            return False
    
    def fetch_ohlcv_data(self, timeframe: str, limit: int = None, 
                        since: Optional[int] = None, refresh_mode: str = 'auto') -> pd.DataFrame:
        """
        获取历史K线数据，支持智能增量更新
        
        参数:
            timeframe: 时间周期
            limit: 获取数据的条数限制
            since: 开始时间戳(毫秒)
            refresh_mode: 刷新模式 - 'auto'(智能增量), 'full'(全量刷新), 'cache_only'(仅使用缓存)
            
        返回:
            DataFrame: 包含OHLCV数据的DataFrame
        """
        # 加载缓存数据和元数据
        cached_df, metadata = self.load_cached_data(timeframe)
        
        # 如果是仅使用缓存模式，直接返回
        if refresh_mode == 'cache_only':
            if cached_df is None:
                self.logger.warning("请求仅使用缓存，但缓存不存在")
            return cached_df
        
        # 检查是否需要刷新数据
        needs_refresh = True
        if refresh_mode == 'auto' and cached_df is not None and 'last_updated' in metadata:
            try:
                # 确保last_updated是字符串类型
                if isinstance(metadata['last_updated'], str):
                    last_updated = datetime.fromisoformat(metadata['last_updated'])
                    elapsed_hours = (datetime.now() - last_updated).total_seconds() / 3600
                    
                    # 根据时间周期决定刷新间隔
                    refresh_hours = {
                        '1m': 0.5,    # 30分钟
                        '5m': 1,      # 1小时
                        '15m': 2,     # 2小时
                        '30m': 3,     # 3小时
                        '1h': 4,      # 4小时
                        '4h': 8,      # 8小时
                        '8h': 12,     # 12小时
                        '1d': 24,     # 24小时
                        '1w': 168,    # 7天
                    }.get(timeframe, 8)  # 默认8小时
                    
                    needs_refresh = elapsed_hours >= refresh_hours
                    if not needs_refresh:
                        self.logger.info(f"{timeframe}数据缓存仍然新鲜 (更新于{elapsed_hours:.1f}小时前)")
                        return cached_df
                    else:
                        self.logger.info(f"{timeframe}数据缓存已过期 (更新于{elapsed_hours:.1f}小时前)")
                else:
                    self.logger.warning(f"元数据中last_updated不是字符串类型: {type(metadata['last_updated'])}")
                    
            except (ValueError, KeyError, TypeError) as e:
                self.logger.warning(f"解析上次更新时间出错: {e}, 将进行数据刷新")
        
        # 如果没有交易所连接或无需刷新，返回缓存数据
        if self.exchange is None:
            self.logger.error("交易所连接不可用，无法获取数据")
            return cached_df
        
        # 确定增量获取的起始时间
        if refresh_mode == 'auto' and cached_df is not None and 'last_timestamp' in metadata:
            try:
                # 确保last_timestamp是字符串类型
                if isinstance(metadata['last_timestamp'], str):
                    # 如果有缓存数据，获取最后一条数据的时间戳，再往前取一些以确保重叠
                    last_cached_time = datetime.fromisoformat(metadata['last_timestamp'])
                    
                    # 根据时间周期计算重叠的时间（获取比最后时间更早一点的数据以确保无缝衔接）
                    overlap_periods = {
                        '1m': 10,
                        '5m': 10,
                        '15m': 8,
                        '30m': 6, 
                        '1h': 5,
                        '4h': 3,
                        '8h': 2,
                        '1d': 2,
                        '1w': 1,
                    }.get(timeframe, 5)
                    
                    # 计算重叠时间
                    if timeframe.endswith('m'):
                        minutes = int(timeframe[:-1]) * overlap_periods
                        overlap_time = timedelta(minutes=minutes)
                    elif timeframe.endswith('h'):
                        hours = int(timeframe[:-1]) * overlap_periods
                        overlap_time = timedelta(hours=hours)
                    elif timeframe.endswith('d'):
                        days = int(timeframe[:-1]) * overlap_periods
                        overlap_time = timedelta(days=days)
                    elif timeframe.endswith('w'):
                        weeks = int(timeframe[:-1]) * overlap_periods
                        overlap_time = timedelta(weeks=weeks)
                    else:
                        overlap_time = timedelta(days=1)
                    
                    # 设置获取数据的起始时间
                    since_time = last_cached_time - overlap_time
                    since_timestamp = int(since_time.timestamp() * 1000)
                    self.logger.info(f"从 {since_time.isoformat()} 开始增量获取数据")
                    since = since_timestamp
                else:
                    self.logger.warning(f"元数据中last_timestamp不是字符串类型: {type(metadata['last_timestamp'])}")
                    since = None
                
            except (ValueError, KeyError, TypeError) as e:
                self.logger.warning(f"计算增量获取起始时间出错: {e}, 将进行全量获取")
                since = None
        
        # 如果是全量刷新，则不设置since
        if refresh_mode == 'full':
            since = None
            
        # 设置重试参数
        max_retries = 3
        retry_delay = 2  # 秒
        
        # 获取新数据
        new_df = None
        for attempt in range(max_retries):
            try:
                self.logger.info(f"从{self.exchange_id}获取{self.symbol} {timeframe}数据 (尝试 {attempt+1}/{max_retries})...")
                
                # 获取OHLCV数据
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol, 
                    timeframe=timeframe,
                    limit=limit,
                    since=since
                )
                
                if not ohlcv or len(ohlcv) == 0:
                    self.logger.warning(f"获取到的数据为空，尝试 {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))
                        continue
                    else:
                        self.logger.error("多次尝试后仍未获取到数据")
                        return cached_df
                
                # 转换为DataFrame
                new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                
                # 确保数据按时间排序
                new_df = new_df.sort_values('timestamp')
                break
                
            except ccxt.NetworkError as e:
                sleep_time = retry_delay * (2 ** attempt)
                self.logger.warning(f"网络错误 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("达到最大重试次数，使用缓存数据")
                    return cached_df
                    
            except Exception as e:
                self.logger.error(f"获取{timeframe}数据出错: {type(e).__name__}: {e}")
                return cached_df
        
        # 合并新旧数据
        if new_df is not None and not new_df.empty:
            if cached_df is not None and not cached_df.empty and refresh_mode == 'auto':
                # 删除重复数据（基于时间戳）
                combined_df = pd.concat([cached_df, new_df])
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp')
                
                # 记录新增了多少条数据
                added_rows = len(combined_df) - len(cached_df)
                self.logger.info(f"增量更新: 新增{added_rows}条数据")
                
                # 更新元数据
                metadata['last_updated'] = datetime.now().isoformat()
                
                # 保存合并后的数据
                self.save_data_and_metadata(timeframe, combined_df, metadata)
                return combined_df
            else:
                # 保存新数据
                metadata = {
                    'last_updated': datetime.now().isoformat(),
                    'last_timestamp': new_df['timestamp'].max().isoformat(),
                    'first_timestamp': new_df['timestamp'].min().isoformat(),
                    'count': len(new_df)
                }
                self.save_data_and_metadata(timeframe, new_df, metadata)
                return new_df
        else:
            return cached_df


class IndicatorCalculator:
    """
    技术指标计算器
    
    负责计算各种技术指标，支持自定义参数和指标组合
    """
    
    def __init__(self, logger: logging.Logger):
        """
        初始化技术指标计算器
        
        参数:
            logger: 日志记录器
        """
        self.logger = logger
        
        # 默认指标参数
        self.default_params = {
            'MA': {'periods': [5, 10, 20, 50, 100, 200]},
            'EMA': {'periods': [5, 10, 20, 50, 100]}, 
            'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'RSI': {'period': 14},
            'BB': {'period': 20, 'dev_up': 2, 'dev_down': 2},
            'STOCH': {'k_period': 14, 'k_slowing': 3, 'd_period': 3},
            'ADX': {'period': 14},
            'CCI': {'period': 14},
            'OBV': {},
            'ATR': {'period': 14},
            'KDJ': {'k_period': 9, 'd_period': 3, 'j_period': 3},
            'VWAP': {}
        }
        
    def calculate_indicators(self, df: pd.DataFrame, 
                           indicators: List[str] = None, 
                           params: Dict[str, Dict] = None) -> pd.DataFrame:
        """
        计算指定的技术指标
        
        参数:
            df: 包含OHLCV数据的DataFrame
            indicators: 要计算的指标列表，None表示计算所有支持的指标
            params: 技术指标参数，用于覆盖默认参数
            
        返回:
            DataFrame: 添加了技术指标的DataFrame
        """
        if df is None or df.empty:
            self.logger.error("输入DataFrame为空，无法计算指标")
            return None
            
        # 如果没有指定指标，使用所有指标
        if indicators is None:
            indicators = list(self.default_params.keys())
            
        # 合并自定义参数与默认参数
        indicator_params = {}
        for ind in self.default_params:
            indicator_params[ind] = self.default_params[ind].copy()
        
        if params:
            for ind, ind_params in params.items():
                if ind in indicator_params:
                    indicator_params[ind].update(ind_params)
                    
        # 确保数据类型正确
        try:
            df = df.copy()
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col not in df.columns:
                    self.logger.error(f"缺少关键列: {col}")
                    return df
                    
                # 转换为数值类型
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 检查并修复NaN值
            if df[numeric_columns].isna().any().any():
                self.logger.warning("数据中存在NaN值，将被填充")
                df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
                
        except Exception as e:
            self.logger.error(f"准备数据时出错: {e}")
            return df
            
        # 计算各项指标
        calculation_methods = {
            'MA': self._calculate_ma,
            'EMA': self._calculate_ema,
            'MACD': self._calculate_macd,
            'RSI': self._calculate_rsi,
            'BB': self._calculate_bollinger,
            'STOCH': self._calculate_stochastic,
            'ADX': self._calculate_adx,
            'CCI': self._calculate_cci,
            'OBV': self._calculate_obv,
            'ATR': self._calculate_atr,
            'KDJ': self._calculate_kdj,
            'VWAP': self._calculate_vwap
        }
        
        # 按指定顺序计算指标
        for indicator in indicators:
            if indicator in calculation_methods:
                try:
                    self.logger.debug(f"计算 {indicator} 指标")
                    df = calculation_methods[indicator](df, indicator_params[indicator])
                except Exception as e:
                    self.logger.error(f"计算 {indicator} 指标出错: {e}")
                    
        return df
        
    def _calculate_ma(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算移动平均线"""
        for period in params['periods']:
            try:
                df[f'MA{period}'] = talib.SMA(df['close'].values, timeperiod=period)
            except Exception as e:
                self.logger.error(f"计算MA{period}出错: {e}")
        return df
        
    def _calculate_ema(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算指数移动平均线"""
        for period in params['periods']:
            try:
                df[f'EMA{period}'] = talib.EMA(df['close'].values, timeperiod=period)
            except Exception as e:
                self.logger.error(f"计算EMA{period}出错: {e}")
        return df
        
    def _calculate_macd(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算MACD指标"""
        try:
            macd, signal, hist = talib.MACD(
                df['close'].values,
                fastperiod=params['fast_period'],
                slowperiod=params['slow_period'],
                signalperiod=params['signal_period']
            )
            df['MACD'] = macd
            df['MACD_signal'] = signal
            df['MACD_hist'] = hist
        except Exception as e:
            self.logger.error(f"计算MACD出错: {e}")
        return df
        
    def _calculate_rsi(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算RSI指标"""
        try:
            df['RSI'] = talib.RSI(df['close'].values, timeperiod=params['period'])
        except Exception as e:
            self.logger.error(f"计算RSI出错: {e}")
        return df
        
    def _calculate_bollinger(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算布林带"""
        try:
            upper, middle, lower = talib.BBANDS(
                df['close'].values,
                timeperiod=params['period'],
                nbdevup=params['dev_up'],
                nbdevdn=params['dev_down'],
                matype=0
            )
            df['BB_upper'] = upper
            df['BB_middle'] = middle
            df['BB_lower'] = lower
        except Exception as e:
            self.logger.error(f"计算布林带出错: {e}")
        return df
        
    def _calculate_stochastic(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算随机指标"""
        try:
            slowk, slowd = talib.STOCH(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                fastk_period=params['k_period'],
                slowk_period=params['k_slowing'],
                slowk_matype=0,
                slowd_period=params['d_period'],
                slowd_matype=0
            )
            df['STOCH_K'] = slowk
            df['STOCH_D'] = slowd
        except Exception as e:
            self.logger.error(f"计算随机指标出错: {e}")
        return df
        
    def _calculate_adx(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算ADX指标"""
        try:
            df['ADX'] = talib.ADX(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=params['period']
            )
        except Exception as e:
            self.logger.error(f"计算ADX出错: {e}")
        return df
        
    def _calculate_cci(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算CCI指标"""
        try:
            df['CCI'] = talib.CCI(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=params['period']
            )
        except Exception as e:
            self.logger.error(f"计算CCI出错: {e}")
        return df
        
    def _calculate_obv(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算OBV指标"""
        try:
            df['OBV'] = talib.OBV(df['close'].values, df['volume'].values)
        except Exception as e:
            self.logger.error(f"计算OBV出错: {e}")
        return df
        
    def _calculate_atr(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算ATR指标"""
        try:
            df['ATR'] = talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=params['period']
            )
        except Exception as e:
            self.logger.error(f"计算ATR出错: {e}")
        return df
        
    def _calculate_kdj(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算KDJ指标"""
        try:
            # 先计算随机震荡指标
            k_period = params['k_period']
            slowk, slowd = talib.STOCH(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                fastk_period=k_period,
                slowk_period=params['d_period'],
                slowk_matype=0,
                slowd_period=params['j_period'],
                slowd_matype=0
            )
            df['KDJ_K'] = slowk
            df['KDJ_D'] = slowd
            # 计算J值: 3*K-2*D
            df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
        except Exception as e:
            self.logger.error(f"计算KDJ出错: {e}")
        return df
        
    def _calculate_vwap(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """计算成交量加权平均价格"""
        try:
            # 确保DataFrame有日期索引以计算交易日
            df_copy = df.copy()
            if 'timestamp' in df_copy.columns:
                df_copy['date'] = df_copy['timestamp'].dt.date
            else:
                df_copy['date'] = pd.to_datetime(df_copy.index).date
                
            # 对每个交易日分别计算VWAP
            df_copy['vwap_numerator'] = df_copy['close'] * df_copy['volume']
            df_copy['vwap_denominator'] = df_copy['volume']
            
            # 按日期分组并累计计算
            groups = df_copy.groupby('date')
            df_copy['vwap_numerator_cum'] = groups['vwap_numerator'].cumsum()
            df_copy['vwap_denominator_cum'] = groups['vwap_denominator'].cumsum()
            
            # 计算VWAP
            df_copy['VWAP'] = df_copy['vwap_numerator_cum'] / df_copy['vwap_denominator_cum']
            
            # 将VWAP复制回原始DataFrame
            df['VWAP'] = df_copy['VWAP']
            
            # 移除临时列
            df_copy.drop(['vwap_numerator', 'vwap_denominator', 
                         'vwap_numerator_cum', 'vwap_denominator_cum'], axis=1, inplace=True)
                         
        except Exception as e:
            self.logger.error(f"计算VWAP出错: {e}")
        return df


class TrendAnalyzer:
    """
    趋势分析器
    
    分析价格和技术指标，评估市场趋势
    """
    
    def __init__(self, logger: logging.Logger):
        """
        初始化趋势分析器
        
        参数:
            logger: 日志记录器
        """
        self.logger = logger
        
    def analyze_trend(self, df: pd.DataFrame, timeframe: str, 
                     timeframe_name: str = None) -> Dict[str, Any]:
        """
        分析价格趋势
        
        参数:
            df: 包含OHLCV和技术指标的DataFrame
            timeframe: 时间周期代码
            timeframe_name: 时间周期的显示名称
            
        返回:
            dict: 趋势分析结果
        """
        default_result = {
            'trend': '未知', 
            'strength': 0, 
            'summary': f"数据不足，无法分析 {timeframe_name or timeframe} 趋势"
        }
        
        if df is None or len(df) < 10:
            self.logger.error(f"{timeframe} 分析: 数据不足")
            return default_result
            
        # 检查必要的技术指标列是否存在
        required_basic_columns = ['close', 'high', 'low', 'volume']
        missing_basic_columns = [col for col in required_basic_columns if col not in df.columns]
        
        if missing_basic_columns:
            self.logger.error(f"{timeframe} 分析: 缺少基础列: {missing_basic_columns}")
            return default_result
            
        try:
            # 配置趋势分析因子
            factors = self._calculate_trend_factors(df, timeframe)
            
            # 计算趋势得分
            trend_score = sum(factors.values())
            
            # 根据得分判断趋势
            trend, strength = self._score_to_trend(trend_score)
            
            # 生成分析摘要
            summary = self._generate_trend_summary(df, timeframe, timeframe_name, trend, factors)
            
            # 计算价格统计
            price_stats = self._calculate_price_statistics(df)
            
            # 构建完整的分析结果
            result = {
                'trend': trend,
                'strength': strength,
                'score': trend_score,
                'summary': summary,
                'factors': factors,
                **price_stats
            }
            
            # 添加技术指标状态
            indicator_status = self._get_indicator_status(df)
            result.update(indicator_status)
            
            return result
            
        except Exception as e:
            self.logger.error(f"{timeframe} 趋势分析出错: {type(e).__name__}: {e}")
            return default_result
            
    def _calculate_trend_factors(self, df: pd.DataFrame, timeframe: str) -> Dict[str, int]:
        """计算趋势因子"""
        factors = {}
        current = df.iloc[-1]
        
        # 确保有至少两个数据点进行比较
        if len(df) < 2:
            self.logger.warning(f"{timeframe} 分析: 数据点数量过少，无法可靠地计算趋势因子")
            return {'insufficient_data': 0}
            
        prev = df.iloc[-2]
        
        # 1. 价格相对于移动平均线
        if all(col in df.columns for col in ['MA20', 'MA50']):
            factors['price_vs_ma20'] = 1 if current['close'] > current['MA20'] else -1
            factors['price_vs_ma50'] = 1 if current['close'] > current['MA50'] else -1
            
            # 价格相对于更长期均线
            if 'MA100' in df.columns:
                factors['price_vs_ma100'] = 1 if current['close'] > current['MA100'] else -1
            if 'MA200' in df.columns:
                factors['price_vs_ma200'] = 1 if current['close'] > current['MA200'] else -1
        
        # 2. 移动平均线排列
        if all(col in df.columns for col in ['MA5', 'MA10', 'MA20']):
            if not pd.isna(current['MA5']) and not pd.isna(current['MA10']) and not pd.isna(current['MA20']):
                ma_aligned_bullish = (current['MA5'] > current['MA10']) and (current['MA10'] > current['MA20'])
                ma_aligned_bearish = (current['MA5'] < current['MA10']) and (current['MA10'] < current['MA20'])
                factors['ma_alignment'] = 1 if ma_aligned_bullish else -1 if ma_aligned_bearish else 0
        
        # 3. MACD
        if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_hist']):
            if not pd.isna(current['MACD']) and not pd.isna(current['MACD_signal']):
                factors['macd_signal'] = 1 if current['MACD'] > current['MACD_signal'] else -1
            
            if not pd.isna(current['MACD_hist']):
                factors['macd_hist'] = 1 if current['MACD_hist'] > 0 else -1
                
                if not pd.isna(prev['MACD_hist']):
                    factors['macd_hist_change'] = 1 if current['MACD_hist'] > prev['MACD_hist'] else -1
        
        # 4. RSI
        if 'RSI' in df.columns:
            if not pd.isna(current['RSI']) and not pd.isna(prev['RSI']):
                factors['rsi'] = 1 if current['RSI'] > 50 else -1
                factors['rsi_trend'] = 1 if current['RSI'] > prev['RSI'] else -1
                
                # 超买/超卖条件
                if current['RSI'] > 70:
                    factors['rsi_overbought'] = -2  # 超买通常是卖出信号
                elif current['RSI'] < 30:
                    factors['rsi_oversold'] = 2     # 超卖通常是买入信号
        
        # 5. 价格趋势
        lookback = min(5, len(df))
        recent_prices = df['close'].tail(lookback).values
        factors['price_trend'] = 1 if recent_prices[-1] > recent_prices[0] else -1
        
        # 短期价格趋势（最近3个周期）
        if len(df) >= 4:
            short_trend_prices = df['close'].tail(3).values
            factors['short_price_trend'] = 1 if short_trend_prices[-1] > short_trend_prices[0] else -1
        
        # 6. 布林带位置
        if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            if not pd.isna(current['BB_upper']) and not pd.isna(current['BB_lower']) and not pd.isna(current['BB_middle']):
                if current['close'] > current['BB_upper']:
                    factors['bb_position'] = -2  # 超买
                elif current['close'] < current['BB_lower']:
                    factors['bb_position'] = 2  # 超卖
                elif current['close'] > current['BB_middle']:
                    factors['bb_position'] = 1  # 偏强
                else:
                    factors['bb_position'] = -1  # 偏弱
                    
                # 布林带宽度变化
                if len(df) >= 10:
                    recent_bandwidth = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
                    if not recent_bandwidth.isna().all():
                        current_bw = recent_bandwidth.iloc[-1]
                        prev_bw = recent_bandwidth.iloc[-6]
                        factors['bb_expansion'] = 1 if current_bw > prev_bw else -1
        
        # 7. ADX趋势强度
        if 'ADX' in df.columns:
            if not pd.isna(current['ADX']):
                if current['ADX'] > 25:
                    # 强趋势
                    factors['trend_strength'] = 2 if factors.get('price_trend', 0) == 1 else -2
                else:
                    # 弱趋势
                    factors['trend_strength'] = 1 if factors.get('price_trend', 0) == 1 else -1
        
        # 8. 随机指标
        if all(col in df.columns for col in ['STOCH_K', 'STOCH_D']):
            if not pd.isna(current['STOCH_K']) and not pd.isna(current['STOCH_D']):
                # K线和D线关系
                factors['stoch_kd'] = 1 if current['STOCH_K'] > current['STOCH_D'] else -1
                
                # 超买超卖
                if current['STOCH_K'] > 80 and current['STOCH_D'] > 80:
                    factors['stoch_overbought'] = -2
                elif current['STOCH_K'] < 20 and current['STOCH_D'] < 20:
                    factors['stoch_oversold'] = 2
        
        # 9. CCI 指标
        if 'CCI' in df.columns:
            if not pd.isna(current['CCI']):
                factors['cci'] = 1 if current['CCI'] > 0 else -1
                
                # 超买超卖
                if current['CCI'] > 200:
                    factors['cci_overbought'] = -2
                elif current['CCI'] < -200:
                    factors['cci_oversold'] = 2
                    
        # 10. OBV 指标 - 成交量是否支撑价格趋势
        if 'OBV' in df.columns:
            if not pd.isna(current['OBV']) and len(df) > 5:
                obv_5_ago = df['OBV'].iloc[-6]
                price_5_ago = df['close'].iloc[-6]
                
                price_change = (current['close'] - price_5_ago) / price_5_ago
                obv_change = (current['OBV'] - obv_5_ago) / obv_5_ago if obv_5_ago != 0 else 0
                
                if price_change > 0 and obv_change > 0:
                    # 价格上涨且成交量支撑
                    factors['obv_confirms_trend'] = 2
                elif price_change < 0 and obv_change < 0:
                    # 价格下跌且成交量支撑
                    factors['obv_confirms_trend'] = -2
                elif price_change > 0 > obv_change:
                    # 价格上涨但成交量不支持，可能是假突破
                    factors['obv_divergence'] = -1
                elif price_change < 0 < obv_change:
                    # 价格下跌但成交量不支持，可能是假跌破
                    factors['obv_divergence'] = 1
        
        # 11. KDJ 指标
        if all(col in df.columns for col in ['KDJ_K', 'KDJ_D', 'KDJ_J']):
            if not pd.isna(current['KDJ_J']):
                # J值偏离程度
                factors['kdj_j'] = 1 if current['KDJ_J'] > 50 else -1
                
                # 超买超卖
                if current['KDJ_J'] > 100:
                    factors['kdj_overbought'] = -2
                elif current['KDJ_J'] < 0:
                    factors['kdj_oversold'] = 2
                    
                # KDJ交叉
                if len(df) >= 2:
                    prev_k = df['KDJ_K'].iloc[-2]
                    prev_d = df['KDJ_D'].iloc[-2]
                    curr_k = current['KDJ_K']
                    curr_d = current['KDJ_D']
                    
                    if prev_k < prev_d and curr_k > curr_d:
                        # 金叉
                        factors['kdj_golden_cross'] = 2
                    elif prev_k > prev_d and curr_k < curr_d:
                        # 死叉
                        factors['kdj_death_cross'] = -2
        
        # 12. VWAP 位置
        if 'VWAP' in df.columns:
            if not pd.isna(current['VWAP']):
                factors['price_vs_vwap'] = 1 if current['close'] > current['VWAP'] else -1
                
        return factors
            
    def _score_to_trend(self, trend_score: float) -> Tuple[str, int]:
        """将趋势得分转换为趋势描述和强度"""
        if trend_score >= 8:
            trend = '强烈上涨'
            strength = 3
        elif trend_score >= 4:
            trend = '上涨'
            strength = 2
        elif trend_score > 0:
            trend = '小幅上涨'
            strength = 1
        elif trend_score == 0:
            trend = '盘整'
            strength = 0
        elif trend_score > -4:
            trend = '小幅下跌'
            strength = -1
        elif trend_score > -8:
            trend = '下跌'
            strength = -2
        else:
            trend = '强烈下跌'
            strength = -3
            
        return trend, strength
        
    def _calculate_price_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算价格相关的统计数据"""
        current = df.iloc[-1]
        
        # 计算各种时间范围的价格变化
        stats = {'price': current['close']}
        
        try:
            # 近期涨跌幅
            lookbacks = [
                (5, '5_period_change'), 
                (10, '10_period_change'),
                (20, '20_period_change')
            ]
            
            for periods, key in lookbacks:
                if len(df) > periods:
                    price_change = ((current['close'] - df['close'].iloc[-periods-1]) / df['close'].iloc[-periods-1]) * 100
                    stats[key] = price_change
            
            # 波动性
            if len(df) >= 5:
                price_volatility = (df['high'].tail(5).max() - df['low'].tail(5).min()) / df['close'].iloc[-5] * 100
                stats['volatility'] = price_volatility
                
            # 均值回归指标 - 当前价格与N日均价的偏离程度
            if 'MA20' in df.columns and not pd.isna(current['MA20']):
                deviation = ((current['close'] - current['MA20']) / current['MA20']) * 100
                stats['ma20_deviation'] = deviation
                
        except Exception as e:
            self.logger.error(f"计算价格统计出错: {e}")
            
        return stats
        
    def _get_indicator_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取技术指标的状态描述"""
        results = {}
        current = df.iloc[-1]
        
        try:
            # MACD状态
            if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_hist']):
                if not pd.isna(current['MACD']) and not pd.isna(current['MACD_signal']) and not pd.isna(current['MACD_hist']):
                    if len(df) > 1:
                        prev = df.iloc[-2]
                        macd_status = "金叉" if current['MACD'] > current['MACD_signal'] and prev['MACD'] <= prev['MACD_signal'] else \
                                     "死叉" if current['MACD'] < current['MACD_signal'] and prev['MACD'] >= prev['MACD_signal'] else \
                                     "多头" if current['MACD'] > current['MACD_signal'] else "空头"
                    else:
                        macd_status = "多头" if current['MACD'] > current['MACD_signal'] else "空头"
                    
                    results['macd_status'] = macd_status
                    results['macd_value'] = current['MACD']
                    results['macd_signal_value'] = current['MACD_signal']
                    results['macd_hist_value'] = current['MACD_hist']
            
            # RSI状态
            if 'RSI' in df.columns and not pd.isna(current['RSI']):
                rsi_value = current['RSI']
                rsi_status = "超买" if rsi_value > 70 else \
                            "超卖" if rsi_value < 30 else \
                            "多头" if rsi_value > 50 else "空头"
                            
                results['rsi_value'] = rsi_value
                results['rsi_status'] = rsi_status
                
            # 布林带状态
            if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
                if not pd.isna(current['BB_upper']) and not pd.isna(current['BB_lower']):
                    bb_position = "上轨上方" if current['close'] > current['BB_upper'] else \
                                "下轨下方" if current['close'] < current['BB_lower'] else \
                                "上轨与中轨之间" if current['close'] >= current['BB_middle'] else \
                                "下轨与中轨之间"
                                
                    bb_width = (current['BB_upper'] - current['BB_lower']) / current['BB_middle'] if current['BB_middle'] != 0 else 0
                    results['bb_position'] = bb_position
                    results['bb_width'] = bb_width
            
            # KDJ状态
            if all(col in df.columns for col in ['KDJ_K', 'KDJ_D', 'KDJ_J']):
                if not pd.isna(current['KDJ_J']):
                    kdj_status = "超买区" if current['KDJ_J'] > 100 else \
                                "超卖区" if current['KDJ_J'] < 0 else \
                                "多头" if current['KDJ_J'] > 50 else "空头"
                                
                    results['kdj_j_value'] = current['KDJ_J']
                    results['kdj_status'] = kdj_status
                
        except Exception as e:
            self.logger.error(f"获取指标状态出错: {e}")
            
        return results
            
    def _generate_trend_summary(self, df: pd.DataFrame, timeframe: str, 
                              timeframe_name: str, trend: str, 
                              factors: Dict[str, int]) -> str:
        """生成趋势分析摘要"""
        if timeframe_name is None:
            timeframe_name = timeframe
            
        current = df.iloc[-1]
        
        try:
            # 基本信息
            summary = [f"BTC {timeframe_name}趋势: {trend}"]
            summary.append(f"价格: {current['close']:.2f} USDT")
            
            # 近期价格变化
            if len(df) > 5:
                price_change = ((current['close'] - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
                summary.append(f"近5个{timeframe_name}涨跌: {price_change:.2f}%")
            
            # 波动性
            if len(df) >= 5:
                price_volatility = (df['high'].tail(5).max() - df['low'].tail(5).min()) / df['close'].iloc[-5] * 100
                summary.append(f"波动幅度: {price_volatility:.2f}%")
            
            # 技术指标状态
            if all(col in df.columns for col in ['MACD', 'MACD_signal']):
                if not pd.isna(current['MACD']) and not pd.isna(current['MACD_signal']):
                    macd_status = "多头" if factors.get('macd_signal', 0) > 0 else "空头"
                    macd_change = "看涨" if factors.get('macd_hist_change', 0) > 0 else "看跌"
                    summary.append(f"MACD: {macd_status}({macd_change})")
                    
            if 'RSI' in df.columns and not pd.isna(current['RSI']):
                rsi_value = current['RSI']
                rsi_status = "超买" if rsi_value > 70 else \
                           "超卖" if rsi_value < 30 else \
                           "多头" if rsi_value > 50 else "空头"
                summary.append(f"RSI({rsi_value:.1f}): {rsi_status}")
                
            # 移动平均线状态
            ma_status = []
            if all(col in df.columns for col in ['MA20', 'MA50']):
                if not pd.isna(current['MA20']) and not pd.isna(current['MA50']):
                    if current['close'] > current['MA20'] and current['close'] > current['MA50']:
                        ma_status.append("价格站上主要均线")
                    elif current['close'] < current['MA20'] and current['close'] < current['MA50']:
                        ma_status.append("价格跌破主要均线")
                    elif current['close'] > current['MA20'] and current['close'] < current['MA50']:
                        ma_status.append("价格处于MA20与MA50之间")
                        
            if ma_status:
                summary.append(f"均线: {', '.join(ma_status)}")
                
            # 额外高级指标
            if 'KDJ_J' in df.columns and not pd.isna(current['KDJ_J']):
                kdj_status = "超买" if current['KDJ_J'] > 100 else \
                           "超卖" if current['KDJ_J'] < 0 else \
                           "偏多" if current['KDJ_J'] > 50 else "偏空"
                summary.append(f"KDJ: {kdj_status} (J={current['KDJ_J']:.1f})")
                
            if 'CCI' in df.columns and not pd.isna(current['CCI']):
                cci_status = "超买" if current['CCI'] > 200 else \
                           "超卖" if current['CCI'] < -200 else \
                           "偏多" if current['CCI'] > 0 else "偏空"
                summary.append(f"CCI: {cci_status} ({current['CCI']:.1f})")
                
            # 返回完整摘要
            return "\n".join(summary)
            
        except Exception as e:
            self.logger.error(f"生成趋势摘要出错: {e}")
            return f"BTC {timeframe_name}趋势: {trend}\n价格: {current['close']:.2f} USDT"


class TrendVisualizer:
    """
    趋势可视化器
    
    负责绘制价格图表和技术指标
    """
    
    def __init__(self, logger: logging.Logger, theme: str = 'default'):
        """
        初始化可视化器
        
        参数:
            logger: 日志记录器
            theme: 图表主题 ('default', 'dark')
        """
        self.logger = logger
        self.theme = theme
        self._setup_style()
        
    def _setup_style(self):
        """设置绘图样式并解决中文显示问题"""
        import platform
        import matplotlib as mpl
        import matplotlib.font_manager as fm
    
        # 检测系统类型
        system_type = platform.system()
        
        # 配置中文字体
        if system_type == 'Darwin':  # macOS
            # 尝试找到Mac系统上可用的中文字体
            self.logger.info("检测到Mac系统，配置中文字体...")
            
            # 获取所有可用字体
            font_files = fm.findSystemFonts()
            chinese_fonts = []
        
            # 常见的Mac中文字体名称关键词
            chinese_keywords = ['heiti', 'hei', 'pingfang', 'hiragino', 'STHeiti', 'microsoft', 'yuanti', 'simsun', 'noto']
            
            # 搜索包含中文关键词的字体
            for font_file in font_files:
                try:
                    font_name = fm.get_font(font_file).family_name.lower()
                    if any(keyword in font_name for keyword in chinese_keywords):
                        chinese_fonts.append(fm.get_font(font_file).family_name)
                except:
                    continue
        
            # 添加常见的Mac中文字体名称
            predefined_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'STHeiti']
            for font in predefined_fonts:
                if font not in chinese_fonts:
                    chinese_fonts.append(font)
        
            if chinese_fonts:
                # 尝试设置找到的字体
                mpl.rcParams['font.family'] = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = chinese_fonts + ['Arial']
                self.logger.info(f"使用以下Mac中文字体: {chinese_fonts[0]}")
            else:
                self.logger.warning("未找到任何中文字体，图表中的中文可能显示为乱码")
                # 使用无衬线字体
                mpl.rcParams['font.family'] = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = ['Arial']
        else:
            # Windows或Linux系统
            mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
        # 解决负号显示问题
        mpl.rcParams['axes.unicode_minus'] = False
        
        # 设置DPI，使图形更清晰
        mpl.rcParams['figure.dpi'] = 100
        
        # 设置主题颜色
        if self.theme == 'dark':
            plt.style.use('dark_background')
            self.line_colors = {
                'price': 'white',
                'ma20': 'orange',
                'ma50': 'cyan',
                'ma100': 'magenta',
                'ma200': 'yellow',
                'bb_upper': 'lime',
                'bb_lower': 'lime',
                'bb_fill': 'green',
                'up': 'green',
                'down': 'red',
            }
        else:
            plt.style.use('default')
            self.line_colors = {
                'price': 'blue',
                'ma20': 'orange',
                'ma50': 'red',
                'ma100': 'purple',
                'ma200': 'black',
                'bb_upper': 'green',
                'bb_lower': 'green',
                'bb_fill': 'lightgreen',
                'up': 'green',
                'down': 'red',
            }  
    def plot_trend(self, dfs: Dict[str, pd.DataFrame], 
                   analysis_results: Dict[str, Dict], 
                   timeframe_names: Dict[str, str], 
                   output_file: str = None) -> None:
        """
        绘制多个时间周期的趋势图
        
        参数:
            dfs: 不同时间周期的DataFrame字典
            analysis_results: 不同时间周期的分析结果字典
            timeframe_names: 时间周期名称字典
            output_file: 输出文件路径，None则显示图表
        """
        timeframes = list(dfs.keys())
        num_timeframes = len(timeframes)
        
        if num_timeframes == 0:
            self.logger.error("没有可用的数据可绘制")
            return
        
        try:
            # 创建子图
            fig, axs = plt.subplots(num_timeframes, 1, figsize=(14, 4 * num_timeframes), dpi=100)
            if num_timeframes == 1:
                axs = [axs]
            
            # 设置标题
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            title = f"BTC多周期趋势分析 - {current_time}"
            fig.suptitle(title, fontsize=16)
            
            for i, timeframe in enumerate(timeframes):
                df = dfs.get(timeframe)
                result = analysis_results.get(timeframe, {})
                tf_name = timeframe_names.get(timeframe, timeframe)
                
                if df is None or len(df) < 10:
                    axs[i].text(0.5, 0.5, f"{tf_name}数据不足", 
                                ha='center', va='center', transform=axs[i].transAxes)
                    continue
                    
                self._plot_single_timeframe(axs[i], df, result, timeframe, tf_name)
            
            # 调整布局
            try:
                plt.tight_layout()
                plt.subplots_adjust(top=0.95)
            except Exception as e:
                self.logger.warning(f"调整图表布局时出错: {e}")
            
            # 保存或显示
            if output_file:
                try:
                    # 确保输出目录存在
                    output_dir = os.path.dirname(output_file)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        
                    plt.savefig(output_file)
                    self.logger.info(f"图表已保存至: {output_file}")
                except Exception as e:
                    self.logger.error(f"保存图表失败: {e}")
            else:
                try:
                    plt.show()
                except Exception as e:
                    self.logger.error(f"显示图表失败: {e}")
            
            # 关闭图表，释放内存
            plt.close()
            
        except Exception as e:
            self.logger.error(f"创建趋势图表时出错: {e}")
    
    def _plot_single_timeframe(self, ax, df, result, timeframe, timeframe_name):
        """绘制单个时间周期的图表"""
        try:
            # 绘制价格
            ax.plot(df['timestamp'], df['close'], label='价格', 
                    color=self.line_colors['price'], linewidth=1.5)
            
            # 绘制移动平均线
            ma_periods = [5, 10, 20, 50, 100, 200]
            for period in ma_periods:
                col = f'MA{period}'
                if col in df.columns and not df[col].isna().all():
                    if period == 20:
                        ax.plot(df['timestamp'], df[col], label=col, 
                                color=self.line_colors['ma20'], linewidth=1)
                    elif period == 50:
                        ax.plot(df['timestamp'], df[col], label=col, 
                                color=self.line_colors['ma50'], linewidth=1)
                    elif period == 100:
                        ax.plot(df['timestamp'], df[col], label=col, 
                                color=self.line_colors['ma100'], linewidth=0.8)
                    elif period == 200:
                        ax.plot(df['timestamp'], df[col], label=col, 
                                color=self.line_colors['ma200'], linewidth=0.8)
                    else:
                        ax.plot(df['timestamp'], df[col], label=col, linewidth=0.8)
            
            # 绘制布林带
            if all(col in df.columns for col in ['BB_upper', 'BB_lower']):
                bb_mask = ~(df['BB_upper'].isna() | df['BB_lower'].isna())
                if bb_mask.any():
                    ax.plot(df.loc[bb_mask, 'timestamp'], df.loc[bb_mask, 'BB_upper'], 
                            color=self.line_colors['bb_upper'], ls='--', alpha=0.3, linewidth=0.8)
                    ax.plot(df.loc[bb_mask, 'timestamp'], df.loc[bb_mask, 'BB_lower'], 
                            color=self.line_colors['bb_lower'], ls='--', alpha=0.3, linewidth=0.8)
                    ax.fill_between(
                        df.loc[bb_mask, 'timestamp'],
                        df.loc[bb_mask, 'BB_upper'],
                        df.loc[bb_mask, 'BB_lower'],
                        alpha=0.1, color=self.line_colors['bb_fill']
                    )
            
            # 设置标题和标签
            trend_color = 'green' if result.get('strength', 0) > 0 else \
                          'red' if result.get('strength', 0) < 0 else 'black'
            
            ax.set_title(
                f"{timeframe_name} 趋势: {result.get('trend', '未知')}", 
                fontsize=14, color=trend_color
            )
            ax.set_ylabel('价格 (USDT)', fontsize=12)
            
            # 格式化X轴日期
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%Y-%m-%d' if timeframe in ['1d', '3d', '1w', '1M'] else '%m-%d %H:%M')
            )
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 旋转X轴标签
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # 添加技术指标标注
            self._add_indicator_annotation(ax, df, result, timeframe)
            
            # 图例
            ax.legend(loc='upper left')
            
        except Exception as e:
            self.logger.error(f"{timeframe} 图表绘制出错: {e}")
            ax.text(0.5, 0.5, f"{timeframe} 图表绘制出错: {str(e)[:50]}...", 
                    ha='center', va='center', transform=ax.transAxes)
    
    def _add_indicator_annotation(self, ax, df, result, timeframe):
        """添加技术指标标注"""
        try:
            # 构建标注文本
            annotations = []
            
            # RSI
            if 'rsi_value' in result and 'rsi_status' in result:
                annotations.append(f"RSI: {result['rsi_value']:.1f} ({result['rsi_status']})")
                
            # MACD
            if 'macd_status' in result:
                annotations.append(f"MACD: {result['macd_status']}")
            
            # 价格变化
            if '5_period_change' in result:
                annotations.append(f"变化: {result['5_period_change']:.2f}%")
            
            # KDJ
            if 'kdj_status' in result and 'kdj_j_value' in result:
                annotations.append(f"KDJ: {result['kdj_status']} (J={result['kdj_j_value']:.1f})")
            
            # 如果有任何标注
            if annotations:
                annotation_text = '\n'.join(annotations)
                
                # 计算标注位置
                y_pos = df['close'].min() + (df['close'].max() - df['close'].min()) * 0.05
                
                ax.annotate(
                    annotation_text,
                    xy=(df['timestamp'].iloc[-1], y_pos),
                    xytext=(-100, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                )
                
        except Exception as e:
            self.logger.error(f"{timeframe} 添加标注时出错: {e}")


class ReportGenerator:
    """
    报告生成器
    
    负责生成趋势分析报告
    """
    
    def __init__(self, logger: logging.Logger):
        """
        初始化报告生成器
        
        参数:
            logger: 日志记录器
        """
        self.logger = logger
        
    def generate_report(self, analysis_results: Dict[str, Dict], 
                       timeframe_names: Dict[str, str], 
                       output_file: str = None) -> str:
        """
        生成综合分析报告
        
        参数:
            analysis_results: 不同时间周期的分析结果字典
            timeframe_names: 时间周期名称字典
            output_file: 输出文件路径
            
        返回:
            str: 报告文本
        """
        # 获取当前时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 创建报告
        report = []
        report.append("=" * 50)
        report.append(f"BTC多时间周期趋势分析报告 - {current_time}")
        report.append("=" * 50)
        report.append("")
        
        # 检查分析结果是否为空
        if not analysis_results:
            report.append("无可用的分析结果。")
            report_text = "\n".join(report)
            
            if output_file:
                self._save_report(report_text, output_file)
            else:
                print(report_text)
                
            return report_text
        
        # 添加各时间周期分析结果
        for timeframe, result in analysis_results.items():
            if 'summary' in result:
                tf_name = timeframe_names.get(timeframe, timeframe)
                report.append(f"【{tf_name}】")
                report.append("-" * 30)
                report.append(result['summary'])
                report.append("")
        
        # 综合趋势分析
        report.append("=" * 50)
        report.append("综合趋势分析")
        report.append("-" * 30)
        
        # 计算不同时间周期的趋势分析
        trend_analysis = self._analyze_multi_timeframe_trends(analysis_results)
        for key, text in trend_analysis.items():
            report.append(text)
            
        # 交易建议
        report.append("")
        report.append("-" * 30)
        report.append("交易建议")
        report.append("-" * 30)
        report.append(self._generate_trading_advice(trend_analysis, analysis_results))
        report.append("")
        
        # 风险提示
        report.append("-" * 30)
        report.append("风险提示")
        report.append("-" * 30)
        report.append("1. 本分析仅供参考，不构成投资建议")
        report.append("2. 加密货币市场波动剧烈，请控制仓位和风险")
        report.append("3. 技术分析不能预测突发事件的影响")
        report.append("4. 始终使用止损单保护您的资金安全")
        
        # 分析时间
        report.append("")
        report.append("-" * 30)
        report.append(f"分析时间: {current_time}")
        report.append("=" * 50)
        
        # 转换为文本
        report_text = "\n".join(report)
        
        # 保存或打印报告
        if output_file:
            self._save_report(report_text, output_file)
        else:
            print(report_text)
        
        return report_text
    
    def _analyze_multi_timeframe_trends(self, analysis_results: Dict[str, Dict]) -> Dict[str, str]:
        """分析多时间周期的趋势"""
        # 获取各时间周期的趋势强度
        strengths = {tf: result.get('strength', 0) for tf, result in analysis_results.items()}
        
        # 按不同范围分类时间周期
        timeframe_categories = {
            'short': ['1m', '5m', '15m', '30m', '1h', '2h', '4h'],
            'medium': ['6h', '8h', '12h', '1d'],
            'long': ['3d', '1w', '1M']
        }
        
        # 计算每个范围的平均趋势强度
        trend_strengths = {}
        
        for category, timeframes in timeframe_categories.items():
            relevant_tfs = [tf for tf in timeframes if tf in strengths]
            if relevant_tfs:
                avg_strength = sum(strengths[tf] for tf in relevant_tfs) / len(relevant_tfs)
                trend_strengths[category] = avg_strength
            else:
                trend_strengths[category] = 0
        
        # 生成趋势描述
        result = {}
        
        for category, strength in trend_strengths.items():
            if category == 'short':
                timeframe_desc = "短期趋势(小时级)"
            elif category == 'medium':
                timeframe_desc = "中期趋势(日内至几日)"
            else:
                timeframe_desc = "长期趋势(周级以上)"
                
            trend_desc = self._strength_to_desc(strength)
            result[category] = f"{timeframe_desc}: {trend_desc}"
            
        return result
        
    def _strength_to_desc(self, strength: float) -> str:
        """将强度值转换为趋势描述"""
        if strength >= 2.5:
            return "强烈看涨"
        elif strength >= 1.5:
            return "看涨"
        elif strength >= 0.5:
            return "略微看涨"
        elif strength > -0.5:
            return "中性"
        elif strength > -1.5:
            return "略微看跌"
        elif strength > -2.5:
            return "看跌"
        else:
            return "强烈看跌"
            
    def _generate_trading_advice(self, trend_analysis: Dict[str, str], 
                               analysis_results: Dict[str, Dict]) -> str:
        """根据趋势分析生成交易建议"""
        try:
            # 提取短中长期趋势强度
            short_term = 0
            medium_term = 0
            long_term = 0
            
            for tf, result in analysis_results.items():
                strength = result.get('strength', 0)
                if tf in ['1m', '5m', '15m', '30m', '1h', '2h', '4h']:
                    short_term += strength
                elif tf in ['6h', '8h', '12h', '1d']:
                    medium_term += strength
                elif tf in ['3d', '1w', '1M']:
                    long_term += strength
            
            # 标准化强度值
            short_count = sum(1 for tf in analysis_results if tf in ['1m', '5m', '15m', '30m', '1h', '2h', '4h'])
            medium_count = sum(1 for tf in analysis_results if tf in ['6h', '8h', '12h', '1d'])
            long_count = sum(1 for tf in analysis_results if tf in ['3d', '1w', '1M'])
            
            short_term = short_term / short_count if short_count > 0 else 0
            medium_term = medium_term / medium_count if medium_count > 0 else 0
            long_term = long_term / long_count if long_count > 0 else 0
            
            # 根据不同时间周期的趋势强度给出建议
            if short_term >= 1 and medium_term >= 1:
                if long_term >= 1:
                    advice = "多头趋势明确，可考虑逢低买入并持有。短、中、长期趋势均向好，适合做多策略。"
                else:
                    advice = "中短期看涨，长期不明确，可考虑短线买入，注意设置止盈。适合波段操作策略，不宜长期持有。"
            elif short_term <= -1 and medium_term <= -1:
                if long_term <= -1:
                    advice = "空头趋势明确，可考虑逢高卖出或持币观望。全周期看跌，市场风险较高。"
                else:
                    advice = "中短期看跌，长期不明确，可考虑减仓或观望。等待市场企稳后再考虑入场。"
            elif abs(short_term) <= 1 and abs(medium_term) <= 1:
                advice = "市场处于盘整状态，建议耐心等待明确信号。此时可采取区间交易策略，但仓位宜轻。"
            elif short_term * medium_term < 0:  # 短期和中期趋势相反
                if abs(short_term) > abs(medium_term):
                    advice = "短期反转信号，但需确认中期趋势配合，建议谨慎。可能是临时反弹/回调，不宜重仓参与。"
                else:
                    advice = "关注短期是否会跟随中期趋势，暂时观望。可设置条件单在趋势一致时入场。"
            else:
                advice = "市场信号混杂，建议等待更明确的趋势确认。此时控制风险为首要考虑因素。"
            
            # 检查技术指标的额外信号
            overbought = False
            oversold = False
            
            for tf in ['1h', '4h', '1d']:
                if tf in analysis_results:
                    result = analysis_results[tf]
                    if result.get('rsi_value', 0) > 70 or 'kdj_overbought' in result.get('factors', {}):
                        overbought = True
                    if result.get('rsi_value', 0) < 30 or 'kdj_oversold' in result.get('factors', {}):
                        oversold = True
            
            if overbought:
                advice += " 注意：部分时间周期出现超买信号，市场可能面临短期回调风险。"
            if oversold:
                advice += " 注意：部分时间周期出现超卖信号，市场可能出现技术性反弹。"
            
            return advice
            
        except Exception as e:
            self.logger.error(f"生成交易建议时出错: {e}")
            return "无法生成详细交易建议，请综合各时间周期趋势自行判断。"
    
    def _save_report(self, report_text: str, output_file: str) -> None:
        """保存报告到文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"报告已保存至: {output_file}")
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")


class BTCTrendAnalyzer:
    """
    比特币多时间周期趋势分析工具
    
    分析不同时间周期的比特币价格趋势，提供技术指标和趋势判断，支持并行处理
    """
    
    def __init__(self, exchange='binance', symbol='BTC/USDT', cache_dir='btc_data', log_level='INFO'):
        """
        初始化比特币趋势分析器
        
        参数:
            exchange (str): 交易所名称，默认为 'binance'
            symbol (str): 交易对，默认为 'BTC/USDT'
            cache_dir (str): 数据缓存目录
            log_level (str): 日志级别
        """
        # 设置日志
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BTCTrendAnalyzer')
        
        self.exchange_id = exchange
        self.symbol = symbol
        self.cache_dir = cache_dir
        
        # 支持的时间周期及其配置
        self.timeframes = {
            '1h': {'name': '1小时', 'limit': 200},
            '4h': {'name': '4小时', 'limit': 150},
            '8h': {'name': '8小时', 'limit': 100},
            '1d': {'name': '日线', 'limit': 90},
            '1w': {'name': '周线', 'limit': 52}
        }
        
        # 验证交易所
        valid_exchanges = ccxt.exchanges
        if exchange not in valid_exchanges:
            raise ValueError(f"无效的交易所: {exchange}")
        
        # 初始化组件
        try:
            self.data_manager = DataManager(exchange, symbol, cache_dir, self.logger)
            self.indicator_calculator = IndicatorCalculator(self.logger)
            self.trend_analyzer = TrendAnalyzer(self.logger)
            self.visualizer = TrendVisualizer(self.logger)
            self.report_generator = ReportGenerator(self.logger)
            
            self.logger.info(f"已初始化比特币趋势分析器, 交易所: {exchange}, 交易对: {symbol}")
        except Exception as e:
            self.logger.error(f"初始化组件失败: {e}")
            raise
    
    def fetch_and_analyze(self, timeframe: str, refresh_mode: str = 'auto', 
                         indicators: List[str] = None, indicator_params: Dict[str, Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        获取数据并分析特定时间周期的趋势
        
        参数:
            timeframe (str): 时间周期
            refresh_mode (str): 刷新模式 ('auto', 'full', 'cache_only')
            indicators (List[str]): 要计算的指标列表
            indicator_params (Dict[str, Dict]): 技术指标参数
            
        返回:
            Tuple[pd.DataFrame, Dict]: DataFrame和分析结果
        """
        try:
            # 获取数据
            limit = self.timeframes.get(timeframe, {}).get('limit', 100)
            df = self.data_manager.fetch_ohlcv_data(timeframe, limit=limit, refresh_mode=refresh_mode)
            
            if df is None or len(df) < 10:
                self.logger.error(f"无法获取足够的{timeframe}数据进行分析")
                return None, {
                    'trend': '未知', 
                    'strength': 0, 
                    'summary': f"数据获取失败或不足，无法分析 {self.timeframes.get(timeframe, {}).get('name', timeframe)} 趋势"
                }
            
            # 计算技术指标
            df_with_indicators = self.indicator_calculator.calculate_indicators(df, indicators, indicator_params)
            
            # 分析趋势
            timeframe_name = self.timeframes.get(timeframe, {}).get('name', timeframe)
            result = self.trend_analyzer.analyze_trend(df_with_indicators, timeframe, timeframe_name)
            
            return df_with_indicators, result
            
        except Exception as e:
            self.logger.error(f"分析{timeframe}趋势时出错: {e}")
            return None, {
                'trend': '错误', 
                'strength': 0, 
                'summary': f"分析出错: {str(e)}"
            }
    
    def analyze_and_plot_all(self, timeframes: List[str] = None, refresh_mode: str = 'auto', 
                           output_file: str = None, indicators: List[str] = None, 
                           indicator_params: Dict[str, Dict] = None) -> Dict[str, Dict]:
        """
        分析并绘制所有指定时间周期的趋势
        
        参数:
            timeframes (List[str]): 要分析的时间周期列表，默认为所有
            refresh_mode (str): 数据刷新模式 ('auto', 'full', 'cache_only')
            output_file (str): 输出图表文件路径
            indicators (List[str]): 要计算的指标列表
            indicator_params (Dict[str, Dict]): 技术指标参数
        
        返回:
            Dict[str, Dict]: 分析结果
        """
        if timeframes is None:
            timeframes = list(self.timeframes.keys())
        else:
            # 验证时间周期
            invalid_timeframes = [tf for tf in timeframes if tf not in self.timeframes]
            if invalid_timeframes:
                self.logger.warning(f"忽略不支持的时间周期: {', '.join(invalid_timeframes)}")
                timeframes = [tf for tf in timeframes if tf in self.timeframes]
        
        if not timeframes:
            self.logger.error("没有有效的时间周期可分析")
            return {}
        
        # 使用并行处理加速数据获取和分析
        dfs = {}
        analysis_results = {}
        
        self.logger.info(f"开始分析 {len(timeframes)} 个时间周期...")
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交所有任务
            future_to_timeframe = {
                executor.submit(self.fetch_and_analyze, timeframe, refresh_mode, indicators, indicator_params): 
                timeframe for timeframe in timeframes
            }
            
            # 收集结果，使用tqdm显示进度
            for future in tqdm(
                concurrent.futures.as_completed(future_to_timeframe), 
                total=len(timeframes), 
                desc="分析时间周期"
            ):
                timeframe = future_to_timeframe[future]
                try:
                    df, result = future.result()
                    dfs[timeframe] = df
                    analysis_results[timeframe] = result
                    
                    tf_name = self.timeframes[timeframe]['name']
                    if 'summary' in result:
                        self.logger.info(f"\n【{tf_name}】分析完成: {result['trend']}")
                    else:
                        self.logger.warning(f"{timeframe} 分析结果中缺少汇总信息")
                        
                except Exception as e:
                    self.logger.error(f"处理 {timeframe} 时发生错误: {e}")
                    dfs[timeframe] = None
                    analysis_results[timeframe] = {
                        'trend': '错误', 
                        'strength': 0, 
                        'summary': f"处理错误: {str(e)}"
                    }
        
        # 如果所有时间周期都没有数据，记录警告
        if all(df is None for df in dfs.values()):
            self.logger.warning("所有时间周期都没有有效数据，可能无法绘制图表")
        
        # 绘制趋势图
        try:
            # 创建时间周期名称字典
            timeframe_names = {tf: info['name'] for tf, info in self.timeframes.items() if tf in dfs}
            
            self.visualizer.plot_trend(dfs, analysis_results, timeframe_names, output_file)
        except Exception as e:
            self.logger.error(f"绘制趋势图表时出错: {e}")
        
        # 返回分析结果
        return analysis_results
    
    def generate_comprehensive_report(self, analysis_results=None, output_file=None, 
                                    timeframes=None, refresh_mode='auto'):
        """
        生成综合分析报告
        
        参数:
            analysis_results (Dict[str, Dict]): 分析结果字典，如果为None则重新分析
            output_file (str): 输出文件路径
            timeframes (List[str]): 要分析的时间周期列表，默认为所有
            refresh_mode (str): 数据刷新模式
            
        返回:
            str: 报告文本
        """
        if analysis_results is None:
            try:
                analysis_results = self.analyze_and_plot_all(timeframes, refresh_mode)
            except Exception as e:
                self.logger.error(f"生成分析结果时出错: {e}")
                analysis_results = {}
        
        # 创建时间周期名称字典
        timeframe_names = {tf: info['name'] for tf, info in self.timeframes.items()}
        
        # 生成报告
        try:
            report_text = self.report_generator.generate_report(
                analysis_results, timeframe_names, output_file
            )
            return report_text
        except Exception as e:
            self.logger.error(f"生成报告时出错: {e}")
            return f"报告生成失败: {e}"


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='BTC多时间周期趋势分析工具')
    parser.add_argument('--exchange', type=str, default='binance', help='交易所 (默认: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对 (默认: BTC/USDT)')
    parser.add_argument('--timeframes', type=str, default='1h,4h,8h,1d,1w', 
                        help='要分析的时间周期，逗号分隔 (默认: 1h,4h,8h,1d,1w)')
    parser.add_argument('--refresh', type=str, choices=['auto', 'full', 'cache_only'], 
                        default='auto', help='数据刷新模式 (默认: auto)')
    parser.add_argument('--report', type=str, default=None, help='保存报告的文件路径')
    parser.add_argument('--chart', type=str, default=None, help='保存图表的文件路径')
    parser.add_argument('--cache-dir', type=str, default='btc_data', help='数据缓存目录 (默认: btc_data)')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别 (默认: INFO)')
    parser.add_argument('--theme', type=str, default='default', 
                        choices=['default', 'dark'],
                        help='图表主题 (默认: default)')
    parser.add_argument('--indicators', type=str, default=None, 
                        help='要计算的技术指标，逗号分隔 (默认: 所有)')
    
    args = parser.parse_args()
    
    # 解析时间周期
    try:
        timeframes = args.timeframes.split(',')
        if not timeframes:
            print("错误: 至少需要指定一个时间周期")
            return
    except Exception as e:
        print(f"解析时间周期出错: {e}")
        return
    
    # 解析指标列表
    indicators = None
    if args.indicators:
        indicators = args.indicators.split(',')
    
    # 创建分析器
    try:
        analyzer = BTCTrendAnalyzer(
            exchange=args.exchange, 
            symbol=args.symbol,
            cache_dir=args.cache_dir,
            log_level=args.log_level
        )
        
        # 设置主题
        analyzer.visualizer.theme = args.theme
        analyzer.visualizer._setup_style()
        
    except Exception as e:
        print(f"创建分析器失败: {e}")
        return
    
    # 分析并生成图表
    try:
        results = analyzer.analyze_and_plot_all(
            timeframes=timeframes, 
            refresh_mode=args.refresh, 
            output_file=args.chart,
            indicators=indicators
        )
    except Exception as e:
        print(f"分析过程出错: {e}")
        results = {}
    
    # 生成报告
    try:
        analyzer.generate_comprehensive_report(results, args.report)
    except Exception as e:
        print(f"生成报告失败: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序遇到未处理的异常: {e}")
