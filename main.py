import torch
import torchvision
from PIL import Image
from src.utils import get_outputs
from torchvision.transforms import transforms as transforms


class Node:

    def __init__(self, nid, c='object', F=None, N=None):
        self.nid = nid
        self.c = c
        self.p = []
        self.F = F
        self.N = N
        self.p_node = {}
        self.d_nodes = {}
        self.is_plural = False
        self.mask = []
        self.box = []


class UnCoRd:

    def __init__(self):
        self.list_of_nodes = {}

    def _NMT_seq2seq(self, question):
        '''
        Google Neural Machine Translation
        returns question's graph sequence
        graph_txt
        '''
        pass

    def _build_graph(self, graph_txt):
        '''
        Returns list of graph nodes
        built from graph_txt sequence
        '''
        nodes_list = [node.strip() for node in graph_txt.split('<NewNode>')[1:]]
        for nid in range(len(nodes_list)):
            node_txt = nodes_list[nid].split()
            node = Node(nid)
            # define object
            node.c = node_txt[1]
            print(node_txt)

            # parsing relations
            if "<rd>" in node_txt or "<rp>" in node_txt:
                if "<rd>" in node_txt:
                    rd = node_txt.index("<rd>")
                    p_rel = ' '.join(node_txt[rd + 1:-1]).strip()
                    p_id = node_txt[-1]
                    node.p_node[p_id] = p_rel.strip()
                if "<rp>" in node_txt:
                    rp = d_id = -1
                    num_rps = node_txt.count("<rp>")
                    for i in range(num_rps):
                        rp = node_txt.index("<rp>", rp + 1)
                        d_rel = ''
                        for word in node_txt[rp + 1:]:
                            if word.isdigit():
                                d_id = word
                                break
                            else:
                                d_rel += word + ' '

                        node.d_nodes[d_id] = d_rel.strip()

                    ''' To-Do: consider situation when one of types of relations or both are absent '''

            # parsing properties
            if "<p>" in node_txt:
                p_text = nodes_list[nid].split("<p>")[1:]
                p_text[-1] = p_text[-1].split('<')[0]
                node.p += [p.strip() for p in p_text]
            print(node.p)

            # findig F
            if "<F>" in node_txt:
                node.F = node_txt[node_txt.index("<F>") + 1]

            # finding N
            if "<N>" in node_txt:
                node.N = node_txt[node_txt.index("<N>") + 1]

            # is_plural (???)
            if "<is_plural>" in node_txt:
                node.is_plural = True

            print(node.nid)
            print(node.c)
            print(node.F)
            print(node.p_node)
            print(node.d_nodes)
            self.list_of_nodes[nid] = node

    def get_answer(self, img_path, question):

        graph_txt = self._NMT_seq2seq(question)
        self._build_graph(graph_txt)
        masks, boxes, labels = self._detect_objects(img_path)
        answer = self._get_answer(masks, boxes, labels)

        return answer

    def _get_answer(self, nid, masks, boxes, labels):
        '''
        DFS Traversal (recursive)
        '''
        answer = ''
        cur_node = self.list_of_nodes[nid]
        for i, label in enumerate(labels):
            if cur_node.c == label:
                if cur_node.p:
                    success, answer = self.check_properties(cur_node.nid, masks[i], boxes[i])
                if cur_node.F:
                    answer = self.get_property_F(cur_node.nid, masks[i], boxes[i])
                if not cur_node.p_node and not cur_node.d_nodes:
                    pass
                else:
                    for d_nid in cur_node.d_nodes:
                        success, answer = self.check_relation(cur_node.nid, d_nid, masks[i], boxes[i])
                        if success:
                            self._get_answer(d_nid) # here should be mask of daughter object - but how should it be initialized?
                if success and cur_node.N:
                    pass

        return answer



        pass

    def _detect_objects(self, img_path, threshold=0.965):
        '''
        Mask RCNN instance segmentation
        '''
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                                   num_classes=91)
        # set the computation device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the model on to the computation device and set to eval mode
        model.to(device).eval()

        # transform to convert the image to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).to(device)

        masks, boxes, labels = get_outputs(image, model, threshold)

        return masks, boxes, labels

        '''

        for node in self.list_of_nodes:
            object = node.c
            if object in ('thing', 'item', 'object'):
                pass
            else:
                if object in labels:
                    num_obj = labels.count(object)
                    obj_id = -1
                    if node.is_plural:
                        for i in range(num_obj):
                            obj_id = labels.index(object, obj_id + 1)
                            node.mask.append(masks[obj_id])
                            node.box.append(boxes[obj_id])
                    elif num_obj == 1:
                        obj_id = labels.index(object)
                        node.mask.append(masks[obj_id])
                        node.box.append(boxes[obj_id])
                    else:
                        print(f"There are more objects of class '{object}' than 1.")
                else:
                    print(f"No objects of class '{object}' on the image.")

            return masks, boxes, labels
        '''


with open('./val_questions_ext.graph') as f:
    graph_txt = f.read()

with open('./val_questions_ext.en') as f:
    questions = f.read()

graph_text_list = graph_txt.split('\n')
questions_list = questions.split('\n')
n = 269
question = questions_list[n]
graph_txt = graph_text_list[n]
model = UnCoRd()
print(model._build_graph(graph_txt))
print(model._detect_objects('./images/i.webp'))
