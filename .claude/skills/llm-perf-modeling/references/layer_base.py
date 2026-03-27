class Layer_Base:
    def __init__(self, hardware_config, model_config, deploy_config, quant_config):
        # 赋值
        self.hardware_config = hardware_config
        # ...

        # 提炼出算子计算需要的所有变量
        # 比如
        self.ep = self.deploy_config.ep
        # ...

    # 计算总耗时
    def get_cost_time(self):
        if self.is_comm_op:
            return self.comm_time()
        else:
            return self.head_time + max(self.cube_time() + self.vector_time(), self.mem_time())

    # 计算cube时延、访存时延
    # todo

    