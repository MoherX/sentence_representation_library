from module import LstmModel, BilstmModel, CnnModel, SumModel


class ModelFactory:

    def get_model(self, model_name, args, input_size, hidden_size, output_size,
                  vocal_size, embedding_size, dropout_rate, word_embeds):
        if model_name == 'lstm':
            return LstmModel(args, input_size, hidden_size, output_size,
                             vocal_size, embedding_size, dropout_rate, word_embeds)
        elif model_name == 'bilstm':
            return BilstmModel(args, input_size, hidden_size, output_size,
                               vocal_size, embedding_size, dropout_rate, word_embeds)
        elif model_name == 'cnn':
            return CnnModel(args, input_size, hidden_size, output_size,
                            vocal_size, embedding_size, dropout_rate, word_embeds)
        elif model_name == 'sum':
            return SumModel(args, input_size, hidden_size, output_size,
                            vocal_size, embedding_size, dropout_rate, word_embeds)
