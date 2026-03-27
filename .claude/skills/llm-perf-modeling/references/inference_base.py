# 模型推理的基类，主要计算模型整体开销，具体模型有哪些modules在子类中实现

class Inference_Base:
    def __init__(self, hardware_config, model_config, deploy_config, quant_config):
        # 赋值给self

        # 创建modules集合，子类决定有哪些
        self.modules = {}

    # 分别计算各种开销，包括cube、vector、访存和通信
    def get_cube_time(self):
        total_time = 0
        for name, module in self.modules.items():
            total_time += module.get_cube_time()
        return total_time
    

    # 计算输出结果，包括ttft、tpot、吞吐等的呢过
    def get_e2e_profiling(self):
        e2e_pro = {}
        e2e_pro['topt'] = xxx
        # ...
        return e2e_pro