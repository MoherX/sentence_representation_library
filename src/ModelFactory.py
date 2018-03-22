from module import LstmModel, BilstmModel, CnnModel, SumModel


class ModelFactory:

    def get_model(self, data, args):
        if data.HP_encoder_type == 'lstm':
            return LstmModel(args, data)
        elif data.HP_encoder_type == 'bilstm':
            return BilstmModel(args, data)
        elif data.HP_encoder_type == 'cnn':
            return CnnModel(args, data)
        elif data.HP_encoder_type == 'sum':
            return SumModel(args, data)
