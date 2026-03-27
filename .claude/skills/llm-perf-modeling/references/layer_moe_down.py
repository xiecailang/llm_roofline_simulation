class Layer_moe_down(Layer_Base):
    def __init__(self, hardware_config, model_config, deploy_config, quant_config):
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        

    # 计算cube计算量
    def get_cube_flops(self):
        # 每个算子的计算量都不一样，要在这个子类中重写
        pass

    # 计算vector计算量
    def get_vector_flops(self):
        # 每个算子的计算量都不一样，要在这个子类中重写
        pass

    # 计算访存量
    def get_vector_flops(self):
        # 每个算子的访存量都不一样，要在这个子类中重写
        pass

    # 每个算子名称不一样，需要子类修改一下
    def get_profiling(self):
        op_pro = super().get_profiling()
        op_pro['op_name'] = 'moe_down'
    

    