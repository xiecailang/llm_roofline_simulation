"""Inference基类 - 模型推理基类

性能指标计算说明：

核心概念：
- micro_batch_size: 每个TP组处理的batch大小（配置文件中的值）
- total_bs (全局batch): micro_batch_size × num_tp_groups
- num_tp_groups: total_chips / attention_tp / context_parallel
- num_cards: total_chips / chips_per_card
- TPOT: 生成单个token的时间（per-request, ms）
- TTFT: Prefill首token时延（per-request, ms）

Per-Card性能公式：
  Decode: tps_per_card = total_bs / tpot_s / num_cards
          qps_per_card = tps_per_card / output_length
  Prefill: qps_per_card = total_bs / ttft_s / num_cards
           tps_per_card = qps_per_card × input_length

System性能公式：
  system_tps = tps_per_card × num_cards
  system_qps = qps_per_card × num_cards
"""

import csv
import json
from pathlib import Path


class InferenceBase:
    """模型推理基类"""

    def __init__(self, hardware_config, model_config, deploy_config, quant_config):
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.deploy_config = deploy_config
        self.quant_config = quant_config
        self.modules = {}

    def add_module(self, name, module):
        self.modules[name] = module

    def get_cube_time(self):
        return sum(m.get_cube_time() for m in self.modules.values())

    def get_vector_time(self):
        return sum(m.get_vector_time() for m in self.modules.values())

    def get_mem_time(self):
        return sum(m.get_mem_time() for m in self.modules.values())

    def get_comm_time(self):
        return sum(m.get_comm_time() for m in self.modules.values())

    def get_total_time(self):
        return sum(m.get_cost_time() for m in self.modules.values())

    def get_memory_usage(self):
        """计算内存占用 (GB) - 子类实现"""
        return 0.0

    def _get_batch_info(self):
        """计算batch和并行度信息"""
        total_chips = self.hardware_config.total_chips
        chips_per_card = self.hardware_config.chips_per_card
        attention_tp = self.deploy_config.attention_tp
        context_parallel = self.deploy_config.context_parallel
        micro_batch_size = self.deploy_config.micro_batch_size

        num_tp_groups = max(1, total_chips // attention_tp // context_parallel)
        num_cards = max(1, total_chips // chips_per_card)
        total_bs = micro_batch_size * num_tp_groups

        return {
            'total_chips': total_chips,
            'chips_per_card': chips_per_card,
            'attention_tp': attention_tp,
            'context_parallel': context_parallel,
            'num_tp_groups': num_tp_groups,
            'num_cards': num_cards,
            'micro_batch_size': micro_batch_size,
            'total_bs': total_bs,
        }

    def get_e2e_profiling(self):
        """计算端到端性能指标（per-request级别时延）

        PP Pipeline Bubble 处理:
        当 PP > 1 时，pipeline 的有效时间 = max_stage_time + bubble_time
        但在 Roofline 建模中，我们建模的是单个 PP stage 的时间，
        整体 throughput 由 max_stage_time 决定（非最后一个 stage 需要等待最后一个 stage 完成）。

        Pipeline bubble 的影响：
        - 增加 PP 不改变单 stage 的计算时间
        - 但降低了整体 throughput（因为 bubble 时间里芯片空闲）
        - 通过 pipeline_bubble_rate 配置来量化
        """
        total_time = self.get_total_time()
        input_len = self.deploy_config.input_length
        output_len = self.deploy_config.output_length
        mtp_length = self.deploy_config.mtp_length
        mtp_acceptance = self.deploy_config.mtp_acceptance_rate
        micro_batch_size = self.deploy_config.micro_batch_size

        # PP bubble 开销
        pp = self.deploy_config.pipeline_parallel
        if pp > 1:
            bubble_rate = self.deploy_config.pipeline_bubble_rate
            # 有效时间 = stage_time * (1 + bubble_rate)
            # bubble_rate 表示 bubble 占比，例如 0.1 表示额外 10% 的空闲时间
            effective_total_time = total_time * (1 + bubble_rate)
        else:
            effective_total_time = total_time

        is_decode = 'Decode' in self.__class__.__name__
        is_prefill = 'Prefill' in self.__class__.__name__

        if is_decode:
            # MTP: 每次decode步骤产出 (1 + mtp_length * mtp_acceptance) 个token per request
            effective_tokens_per_step = 1.0 + mtp_length * mtp_acceptance if mtp_length > 0 else 1.0
            # TPOT: per-request per-token时间 (ms)
            # total_time已包含batch维度，除以micro_batch_size得到per-request时间
            tpot = effective_total_time / (micro_batch_size * effective_tokens_per_step)
            ttft = None

        elif is_prefill:
            # TTFT: per-request prefill时间 (ms)
            # total_time是处理micro_batch_size个请求的时间
            ttft = effective_total_time / micro_batch_size
            tpot = None

        else:
            # PD混合: ttft为prefill部分, tpot为decode部分
            ttft = effective_total_time / micro_batch_size
            tpot = None

        # 时延分解
        cube_time = self.get_cube_time()
        vector_time = self.get_vector_time()
        mem_time = self.get_mem_time()
        comm_time = self.get_comm_time()

        # 单个transformer block时延（取第一个可用的层）
        single_attn_time = 0.0
        single_ffn_time = 0.0
        for name, module in self.modules.items():
            # 优先找 layer_0，如果不存在则找第一个 attention/ffn
            if 'layer_0_attention' in name:
                single_attn_time = module.get_cost_time()
            elif single_attn_time == 0.0 and '_attention' in name:
                single_attn_time = module.get_cost_time()
            if 'layer_0_moe' in name or 'layer_0_ffn' in name:
                single_ffn_time = module.get_cost_time()
            elif single_ffn_time == 0.0 and ('_moe' in name or '_ffn' in name):
                single_ffn_time = module.get_cost_time()
        single_block_time = single_attn_time + single_ffn_time

        # PP 信息
        pp_stage = getattr(self, 'pp_stage', 0)
        pp_bubble_rate = self.deploy_config.pipeline_bubble_rate if pp > 1 else 0.0

        return {
            'model_name': self.__class__.__name__,
            'total_time_ms': total_time,
            'effective_total_time_ms': effective_total_time,
            'micro_batch_size': micro_batch_size,
            'ttft_ms': ttft,
            'tpot_ms': tpot,
            'output_length': output_len,
            'input_length': input_len,
            'memory_usage_gb': self.get_memory_usage(),
            'single_block_time_ms': single_block_time,
            'single_attn_time_ms': single_attn_time,
            'single_ffn_time_ms': single_ffn_time,
            'pipeline_parallel': pp,
            'pp_stage': pp_stage,
            'pp_bubble_rate': pp_bubble_rate,
            'latency_breakdown': {
                'cube_time_ms': cube_time,
                'vector_time_ms': vector_time,
                'mem_time_ms': mem_time,
                'comm_time_ms': comm_time,
            },
            'latency_ratio': {
                'cube_ratio': cube_time / total_time if total_time > 0 else 0,
                'vector_ratio': vector_time / total_time if total_time > 0 else 0,
                'mem_ratio': mem_time / total_time if total_time > 0 else 0,
                'comm_ratio': comm_time / total_time if total_time > 0 else 0,
            },
            'modules': {name: module.get_profiling() for name, module in self.modules.items()},
        }

    def save_results(self, output_dir: str = "outputs"):
        """保存所有结果文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        profiling = self.get_e2e_profiling()

        # 1. 算子级详细CSV
        self._save_op_details_csv(profiling, output_path / "op_details.csv")

        # 2. 单卡性能JSON
        self._save_single_card_perf(profiling, output_path / "single_card_perf.json")

        # 3. 系统级性能JSON
        self._save_system_perf(profiling, output_path / "system_perf.json")

        return profiling

    def _save_op_details_csv(self, profiling, filepath):
        """保存算子级详细CSV"""
        rows = []
        for module_name, module_data in profiling['modules'].items():
            if 'layers' in module_data:
                for layer_name, layer_data in module_data['layers'].items():
                    rows.append({
                        'module_name': module_name,
                        'op_name': f"{module_name}.{layer_name}",
                        'cube_flops': layer_data.get('cube_flops', 0),
                        'cube_time_ms': layer_data.get('cube_time_ms', 0),
                        'vector_flops': layer_data.get('vector_flops', 0),
                        'vector_time_ms': layer_data.get('vector_time_ms', 0),
                        'mem_bytes': layer_data.get('mem_bytes', 0),
                        'mem_time_ms': layer_data.get('mem_time_ms', 0),
                        'comm_bytes': layer_data.get('comm_bytes', 0),
                        'comm_time_ms': layer_data.get('comm_time_ms', 0),
                        'total_time_ms': layer_data.get('total_time_ms', 0),
                    })

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"  算子详细信息已保存到: {filepath}")

    def _save_single_card_perf(self, profiling, filepath):
        """保存单卡性能JSON（per-card指标）

        公式:
          Decode: tps_per_card = total_bs / tpot_s / num_cards
                  qps_per_card = tps_per_card / output_length
          Prefill: qps_per_card = total_bs / ttft_s / num_cards
                   tps_per_card = qps_per_card × input_length
        """
        batch_info = self._get_batch_info()
        output_len = profiling['output_length']
        input_len = profiling['input_length']
        tpot_ms = profiling['tpot_ms']
        ttft_ms = profiling['ttft_ms']
        num_cards = batch_info['num_cards']
        total_bs = batch_info['total_bs']

        is_decode = tpot_ms is not None
        is_prefill = ttft_ms is not None and tpot_ms is None

        if is_decode:
            tpot_s = tpot_ms / 1000.0
            # tps_per_card = total_bs / tpot_s / num_cards
            tps_per_card = total_bs / tpot_s / num_cards if tpot_s > 0 else 0
            # qps_per_card = tps_per_card / output_length
            qps_per_card = tps_per_card / output_len if output_len > 0 else tps_per_card
        elif is_prefill:
            ttft_s = ttft_ms / 1000.0
            # qps_per_card = total_bs / ttft_s / num_cards
            qps_per_card = total_bs / ttft_s / num_cards if ttft_s > 0 else 0
            # tps_per_card = qps_per_card × input_length
            tps_per_card = qps_per_card * input_len
        else:
            qps_per_card = 0
            tps_per_card = 0

        data = {
            'model_name': profiling['model_name'],
            'total_time_ms': profiling['total_time_ms'],
            'micro_batch_size': profiling['micro_batch_size'],
            'total_bs': total_bs,
            'num_tp_groups': batch_info['num_tp_groups'],
            'num_cards': num_cards,
            'ttft_ms': ttft_ms,
            'tpot_ms': tpot_ms,
            'qps_per_card': qps_per_card,
            'tps_per_card': tps_per_card,
            'output_length': output_len,
            'input_length': input_len,
            'memory_usage_gb': profiling['memory_usage_gb'],
            'single_block_time_ms': profiling['single_block_time_ms'],
            'single_attn_time_ms': profiling['single_attn_time_ms'],
            'single_ffn_time_ms': profiling['single_ffn_time_ms'],
            'latency_breakdown': profiling['latency_breakdown'],
            'latency_ratio': profiling['latency_ratio'],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  单卡性能指标已保存到: {filepath}")

    def _save_system_perf(self, profiling, filepath):
        """保存系统级性能JSON

        system_tps = tps_per_card × num_cards
        system_qps = qps_per_card × num_cards
        """
        batch_info = self._get_batch_info()
        num_cards = batch_info['num_cards']
        total_bs = batch_info['total_bs']
        output_len = profiling['output_length']
        input_len = profiling['input_length']
        tpot_ms = profiling['tpot_ms']
        ttft_ms = profiling['ttft_ms']

        is_decode = tpot_ms is not None
        is_prefill = ttft_ms is not None and tpot_ms is None

        if is_decode:
            tpot_s = tpot_ms / 1000.0
            # system_tps = total_bs / tpot_s (所有卡合计)
            system_tps = total_bs / tpot_s if tpot_s > 0 else 0
            # system_qps = system_tps / output_length
            system_qps = system_tps / output_len if output_len > 0 else system_tps
        elif is_prefill:
            ttft_s = ttft_ms / 1000.0
            # system_qps = total_bs / ttft_s (所有卡合计)
            system_qps = total_bs / ttft_s if ttft_s > 0 else 0
            # system_tps = system_qps × input_length
            system_tps = system_qps * input_len
        else:
            system_qps = 0
            system_tps = 0

        data = {
            'total_chips': batch_info['total_chips'],
            'chips_per_card': batch_info['chips_per_card'],
            'attention_tp': batch_info['attention_tp'],
            'context_parallel': batch_info['context_parallel'],
            'num_tp_groups': batch_info['num_tp_groups'],
            'num_cards': num_cards,
            'micro_batch_size': batch_info['micro_batch_size'],
            'total_bs': total_bs,
            'system_qps': system_qps,
            'system_tps': system_tps,
            'ttft_ms': ttft_ms,
            'tpot_ms': tpot_ms,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  系统级性能指标已保存到: {filepath}")
