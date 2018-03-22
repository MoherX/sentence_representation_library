from module import LstmModel, BilstmModel, CnnModel, SumModel


class ModelFactory:

    def get_model(self, data, args):
        if data.HP_encoder_type == 'lstm':
            return LstmModel(args, data.input_size, data.HP_hidden_dim, data.label_alphabet_size,
                             data.word_alphabet.size(), data.word_emb_dim, data.HP_dropout,
                             data.pretrain_word_embedding)
        elif data.HP_encoder_type == 'bilstm':
            return BilstmModel(args, data.input_size, data.HP_hidden_dim, data.label_alphabet_size,
                               data.word_alphabet.size(), data.word_emb_dim, data.HP_dropout,
                               data.pretrain_word_embedding)
        elif data.HP_encoder_type == 'cnn':
            return CnnModel(args, data.input_size, data.HP_hidden_dim, data.label_alphabet_size,
                            data.word_alphabet.size(), data.word_emb_dim, data.HP_dropout,
                            data.pretrain_word_embedding)
        elif data.HP_encoder_type == 'sum':
            return SumModel(args, data.input_size, data.HP_hidden_dim, data.label_alphabet_size,
                            data.word_alphabet.size(), data.word_emb_dim, data.HP_dropout,
                            data.pretrain_word_embedding)
