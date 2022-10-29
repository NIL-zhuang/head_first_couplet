from operator import ne

from model.t5_base import T5BaseModel

COUPLET_PROMPOT = '对联: '
MAX_SEQ_LEN = 32
MAX_OUT_TOKENS=512

class Couplet():
    def __init__(self, model_path) -> None:
        self.model = T5BaseModel()
        self.model.load_model(model_path, use_gpu=True)
        self.model.model = self.model.model.to('cuda')

    def predict(self, in_str):
        in_request = f"{COUPLET_PROMPOT}{in_str[:MAX_SEQ_LEN]}"
        tgt_len = min(
            MAX_OUT_TOKENS,
            len(in_str[:MAX_SEQ_LEN]) + 2
        )
        return self.model.predict(
            in_request,
            max_length=tgt_len,
            min_length=tgt_len,
            num_beams=1,
            top_p=1.0,
            top_k=1,
            do_sample=True
        )

if __name__=='__main__':
    model_path = r'/home/zhuangzy/head_first_couplet/t5-couplet/simplet5-epoch-4-train-loss-2.4849-val-loss-2.9282'
    model = Couplet(model_path)

    while(True):
        s = input("上联: ")
        next = model.predict(s)
        print("下联: ", next[0][:len(s)])
