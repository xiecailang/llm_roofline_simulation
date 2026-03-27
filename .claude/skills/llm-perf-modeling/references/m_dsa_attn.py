class M_dsa_attention(Moudle_Base):
    def __init__(self, hardware_config, model_config, deploy_config, quant_config):
        # 赋值给self
        super().__init__(hardware_config, model_config, deploy_config, quant_config)
        # 每个module子类来决定有哪些算子
        self.layers = {
            "op_xxx": Layer_xxx(hardware_config, model_config, deploy_config, quant_config)
        }

   