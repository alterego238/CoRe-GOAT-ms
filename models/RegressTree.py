import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np 

from mindspore import context
        
class RegressTree(nn.Cell):
    def __init__(self,in_channel,hidden_channel,depth):
        super(RegressTree, self).__init__()
        self.depth = depth
        self.num_leaf = 2**(depth-1)

        self.first_layer = nn.SequentialCell(
            nn.Dense(in_channel, hidden_channel),
            nn.ReLU()
        )

        self.feature_layers = nn.CellList([self.get_tree_layer(2**d, hidden_channel) for d in range(self.depth - 1)])
        self.clf_layers = nn.CellList([self.get_clf_layer(2**d, hidden_channel) for d in range(self.depth - 1)])
        self.reg_layer = nn.Conv1d(self.num_leaf * hidden_channel, self.num_leaf, 1, group=self.num_leaf, has_bias=True)
    @staticmethod
    def get_tree_layer(num_node_in, hidden_channel=256):
        return nn.SequentialCell(
            nn.Conv1d(num_node_in * hidden_channel, num_node_in * 2 * hidden_channel, 1, group=num_node_in, has_bias=True),
            nn.ReLU()
        )

    @staticmethod
    def get_clf_layer(num_node_in, hidden_channel=256):
        return nn.Conv1d(num_node_in * hidden_channel, num_node_in * 2, 1, group=num_node_in, has_bias=True)

    def construct(self, input_feature):
        out_prob = []
        x = self.first_layer(input_feature)
        bs = x.shape[0]
        x = x.unsqueeze(-1)
        for i in range(self.depth - 1):
            prob = self.clf_layers[i](x).squeeze(-1)
            x = self.feature_layers[i](x)
            # print(prob.shape,x.shape)
            if len(out_prob) > 0:
                prob = ops.log_softmax(prob.view(bs, -1, 2), axis=-1)
                ss = out_prob[-1].shape[1]
                pre_prob = out_prob[-1].view(bs, -1, 1).broadcast_to((bs, ss * 2, 2)).reshape(bs, -1, 2)
                prob = pre_prob + prob
                out_prob.append(prob)
            else:
                out_prob.append(ops.log_softmax(prob.view(bs, -1, 2), axis=-1))  # 2 branch only
        delta = self.reg_layer(x).squeeze(-1)
        # leaf_prob = torch.exp(out_prob[-1].view(bs, -1))
        # assert delta.size() == leaf_prob.size()
        # final_delta = torch.sum(leaf_prob * delta, dim=1)
        return out_prob, delta
      

if __name__ == '__main__':
    from mindspore.common.initializer import One, Normal
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--RT_depth', type=int, default=5, help = '')
    args = parser.parse_args()
    
    regressor = RegressTree(
                        in_channel = 2 * 1024 + 1,
                        hidden_channel = 256, 
                        depth = args.RT_depth)
    X = ms.Tensor(shape=(2, 2049), dtype=ms.float32, init=One())
    print(f'X.shape: {X.shape}')
    out_prob, delta = regressor(X)
    for i, prob in enumerate(out_prob):
        print(f'out_prob[{i}].shape: {prob.shape}')
    print(f'delta.shape:{delta.shape}')
