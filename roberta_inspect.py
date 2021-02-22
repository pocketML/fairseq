import pytorch_inspect as pi
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa # load the Commonsense QA task

roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
roberta.eval() # disable dropout
# roberta.cuda() # using GPU






input = roberta.encode(
                    'Q: ' + "hello",
                    'A: ' + "goodbye",
                    no_separator=True
                )

print(input)
print(input.shape)

pi.summary(roberta, (1, 9, 9)) #(1, 32, 32))