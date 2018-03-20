from module import LstmModel, BilstmModel, CnnModel, SumModel


class ModelFactory:
    def __init__(self, args, input_size, hidden_size, output_size,
                 vocal_size, embedding_size, dropout_rate, word_embeds):
        self.lstm = LstmModel(args, input_size, hidden_size, output_size,
                              vocal_size, embedding_size, dropout_rate, word_embeds)
        self.bilstm = BilstmModel(args, input_size, hidden_size, output_size,
                                  vocal_size, embedding_size, dropout_rate, word_embeds)
        self.cnn = CnnModel(args, input_size, hidden_size, output_size,
                            vocal_size, embedding_size, dropout_rate, word_embeds)
        self.sum = SumModel(args, input_size, hidden_size, output_size,
                            vocal_size, embedding_size, dropout_rate, word_embeds)

    def get_model(self, model_name):
        if model_name == 'lstm':
            return self.lstm
        elif model_name == 'bilstm':
            return self.bilstm
        elif model_name == 'cnn':
            return self.cnn
        elif model_name == 'sum':
            return self.sum
