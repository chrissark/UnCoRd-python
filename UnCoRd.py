import json


class Node:

    def __init__(self, nid, F=None, N=None):
        self.nid = nid
        self.p = {}
        self.F = F
        self.N = N
        self.p_node = {}
        self.d_nodes = {}
        self.is_plural = False


class UnCoRd:

    def __init__(self):
        self.list_of_nodes = {}

    def _NMT_seq2seq(self, question_id):
        '''
        Google Neural Machine Translation
        returns question's graph sequence
        graph_txt

        Temporary it maps the question
        with its qround truth sequence
        '''
        with open('val_questions_none.graph') as f:
            graph_txt = f.read()
        graph_text_list = graph_txt.split('\n')

        return graph_text_list[question_id]

    def _build_graph(self, graph_txt):
        '''
        Returns list of graph nodes
        built from graph_txt sequence
        '''
        COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
        SIZES = ['small', 'large']
        MATERIALS = ['rubber', 'metal']
        SHAPES = ['cube', 'sphere', 'cylinder']
        nodes_list = [node.strip() for node in graph_txt.split('<NewNode>')[1:]]
        for nid in range(len(nodes_list)):
            node_txt = nodes_list[nid].split()
            node = Node(nid + 1)
            # define object
            c = node_txt[1]
            node.p['shape'] = c
            print(node_txt)
            # parsing relations
            if "<rd>" in node_txt or "<rp>" in node_txt:
                if "<rp>" in node_txt:
                    rp = p_id = -1
                    num_rps = node_txt.count("<rp>")
                    for i in range(num_rps):
                        rp = node_txt.index("<rp>", rp + 1)
                        p_rel = ''
                        for word in node_txt[rp + 1:]:
                            if word.isdigit():
                                p_id = word
                                break
                            else:
                                p_rel += word + ' '
                        node.p_node[p_id] = p_rel.strip()

                if "<rd>" in node_txt:
                    rd = d_id = -1
                    num_rds = node_txt.count("<rd>")
                    for i in range(num_rds):
                        rd = node_txt.index("<rd>", rd + 1)
                        d_rel = ''
                        for word in node_txt[rd + 1:]:
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
                properties = [p.strip() for p in p_text]
                for p in properties:
                    if p in COLORS:
                        node.p['color'] = p
                    elif p in SIZES:
                        node.p['size'] = p
                    elif p in MATERIALS:
                        node.p['material'] = p
                    elif p in SHAPES:
                        node.p['shape'] = p

            #print(node.p)

            # findig F
            if "<F>" in node_txt:
                node.F = node_txt[node_txt.index("<F>") + 1]

            # finding N
            if "<N>" in node_txt:
                node.N = node_txt[node_txt.index("<N>") + 1]

            # is_plural (???)
            if "<is_plural>" in node_txt:
                node.is_plural = True

            #print(node.nid)
            #print(node.F)
            #print(node.p_node)
            #print(node.d_nodes)
            self.list_of_nodes[nid + 1] = node

        #print(self.list_of_nodes)

    def get_answer(self, img_path, question):

        if self.list_of_nodes:
            self.list_of_nodes.clear()

        question_id = question['question_index']
        img_id = question['image_index']

        graph_txt = self._NMT_seq2seq(question_id)
        self._build_graph(graph_txt)
        objects, relations = self._detect_objects(img_id)
        answer = self._get_answer(1, objects, relations)

        return answer[1]

    def _get_answer(self, nid, objects, relations, visited_nodes=None, candidate_obj=None):
        '''
        DFS Traversal (recursive)
        '''
        if not visited_nodes:
            visited_nodes = {}
        # print(f'Checking node {nid}')
        answer = ''
        success = False
        cur_node = self.list_of_nodes[nid]
        cur_object = None
        if candidate_obj:
            if cur_node.p:
                success, answer = self.check_properties(cur_node.nid, candidate_obj)
                if success:
                    # print("candidate is suitable")
                    cur_object = candidate_obj
                else:
                    return success, answer
        else:
            for obj in objects:
                if cur_node.p:
                    success, answer = self.check_properties(cur_node.nid, obj)
                if success:
                    cur_object = obj
                    break
        if cur_object:
            visited_nodes[nid] = cur_object
            if cur_node.F:
                success, answer = self.get_property_F(cur_node.nid, cur_object)

            if cur_node.d_nodes or cur_node.p_node:
                #print('Proceeding to daughters...')
                if cur_node.d_nodes:
                    #d_object = None
                    for d_id in cur_node.d_nodes.keys():
                        rel = cur_node.d_nodes[d_id]
                        for obj in objects:
                            obj_id = objects.index(obj)
                            cur_id = objects.index(cur_object)
                            #print(obj_id)
                            #print(cur_id)
                            if obj_id != cur_id:
                                # print('Checking properties...')
                                success, answer = self.check_relations(cur_object, obj, cur_id, obj_id, rel, relations)
                            else:
                                continue

                            if success:
                                if int(d_id) not in visited_nodes:
                                    success, answer = self._get_answer(int(d_id), objects, relations, visited_nodes, obj)
                                    if success:
                                        return success, answer

                            #print(f'answer: {answer}')
                elif cur_node.p_node:
                    for p_id in cur_node.p_node.keys():
                        if int(p_id) not in visited_nodes:
                            rel = cur_node.p_node[p_id]
                            for obj in objects:
                                obj_id = objects.index(obj)
                                cur_id = objects.index(cur_object)
                                # print(obj_id)
                                # print(cur_id)
                                if obj_id != cur_id:
                                    # print('Checking properties...')
                                    success = self.check_relations(obj, cur_object, obj_id, cur_id, rel,
                                                                           relations)[0]
                                else:
                                    continue

                                if success:
                                    success = self.check_properties(int(p_id), obj)[0]

                                if success:
                                    return success, answer
                        if not success:
                            del visited_nodes[nid]
                            return success, answer

            else:
                for i in self.list_of_nodes.keys():
                    if int(i) not in visited_nodes:
                        success, answer = self._get_answer(int(i), objects, relations, visited_nodes)

        return success, answer

    def check_properties(self, nid, obj):

        cur_node = self.list_of_nodes[nid]
        success = False
        answer = 'no'
        for p_key in cur_node.p.keys():
            if p_key != 'shape':
                if cur_node.p[p_key] == obj[p_key]:
                    continue
                else:
                    return success, answer
            else:
                if cur_node.p[p_key] in ('object', 'item', 'thing'):
                    pass
                else:
                    if cur_node.p[p_key] == obj[p_key]:
                        continue
                    else:
                        return success, answer

        success = True
        answer = 'yes'

        return success, answer

    def get_property_F(self, nid, obj):

        cur_node = self.list_of_nodes[nid]
        success = False
        answer = ''

        if cur_node.F in obj.keys():
            answer = obj[cur_node.F]
            success = True

        return success, answer

    def check_relations(self, cur_obj, obj, cur_id, obj_id, rel, relations):

        answer = 'no'
        success = False

        if 'same' in rel:
            print('same check')
            same_p = rel.split()[1]
            if cur_obj[same_p] == obj[same_p]:
                #print(cur_obj)
                #print(obj)
                success = True
                answer = 'yes'
            return success, answer
        else:
            if rel == 'front':
                print('front check')
            elif rel == 'behind':
                print('behind check')
            elif rel == 'right':
                print('right check')
            elif rel == 'left':
                print('left check')
            obj_list = relations[rel][cur_id]
            if obj_id in obj_list:
                print('found')
                success = True
                answer = 'yes'

               # print(cur_obj)
               # print(obj)
            return success, answer

    def _detect_objects(self, img_id):

        with open('CLEVR_val_scenes.json', 'r') as f:
            scenes_info = json.load(f)

        objects = scenes_info['scenes'][img_id]['objects']
        relations = scenes_info['scenes'][img_id]['relationships']

        return objects, relations