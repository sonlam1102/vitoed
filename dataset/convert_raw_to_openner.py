import json
import numpy as np


def extract_text(raw_text, start_indx, end_indx):
    return raw_text[start_indx:end_indx]


def find_expression_entity(entity, lst_entities):
    assert entity['label'] == 'Polarity'
    for e in lst_entities:
        if abs(e['start_offset'] - entity['start_offset']) <= 2 and e['label'] == 'Expression' and abs(e['end_offset'] - entity['end_offset']) <= 2:
            return e
    return None


def find_entity_by_id(ent_id, lst_entities):
    for e in lst_entities:
        if e['id'] == ent_id:
            return e
    return None


def parse_one_opinion(rel, lst_ents, raw_text):
    from_id = rel['from_id']
    to_id = rel['to_id']

    polarity = rel['type']

    from_ent = find_entity_by_id(from_id, lst_ents)
    to_ent = find_entity_by_id(to_id, lst_ents)

    if from_ent['label'] == 'Source':
        source = [
            [extract_text(raw_text, from_ent['start_offset'], from_ent['end_offset'])],
            ["{}:{}".format(from_ent['start_offset'], from_ent['end_offset'])]
        ]
    else:
        source = [
            [],
            []
        ]

    if to_ent['label'] == 'Target':
        target = [
            [extract_text(raw_text, to_ent['start_offset'], to_ent['end_offset'])],
            ["{}:{}".format(to_ent['start_offset'], to_ent['end_offset'])]
        ]
    else:
        target = [
            [],
            []
        ]

    if from_ent['label'] == 'Expression':
        expression = [
            [extract_text(raw_text, from_ent['start_offset'], from_ent['end_offset'])],
            ["{}:{}".format(from_ent['start_offset'], from_ent['end_offset'])]
        ]
    elif to_ent['label'] == 'Expression':
        expression = [
            [extract_text(raw_text, to_ent['start_offset'], to_ent['end_offset'])],
            ["{}:{}".format(to_ent['start_offset'], to_ent['end_offset'])]
        ]
    else:
        if from_ent['label'] == 'Polarity':
            express_ent = find_expression_entity(from_ent, lst_ents)
            expression = [
                [extract_text(raw_text, express_ent['start_offset'], express_ent['end_offset'])],
                ["{}:{}".format(express_ent['start_offset'], express_ent['end_offset'])]
            ] if express_ent is not None else [[], []]
        elif to_ent['label'] == 'Polarity':
            express_ent = find_expression_entity(to_ent, lst_ents)
            expression = [
                [extract_text(raw_text, express_ent['start_offset'], express_ent['end_offset'])],
                ["{}:{}".format(express_ent['start_offset'], express_ent['end_offset'])]
            ] if express_ent is not None else [[], []]
        else:
            expression = [
                [],
                []
            ]

    return source, target, expression, polarity


def is_empty(p):
    if len(p[0]) == 0 and len(p[1]) == 0:
        return True
    return False


def merge_expression(lst_opinions):
    lst_indx_opinions = []
    for lo in lst_opinions:
        if is_empty(lo['Polar_expression']):
            for los in lst_opinions:
                if not is_empty(los['Target']) and los['Target'][1][0] == lo['Target'][1][0]:
                    if not is_empty(los['Polar_expression']):
                        lo['Polar_expression'] = los['Polar_expression']
                        break

    for lo in lst_opinions:
        # if is_empty(lo['Polar_expression']):
        #     continue
        lst_indx_opinions.append(lo['Polar_expression'][1][0])
    lst_indx_opinions = list(set(lst_indx_opinions))

    new_lst_opinions = []
    for li in lst_indx_opinions:
        tmp = []
        for lo in lst_opinions:
            # if is_empty(lo['Polar_expression']):
            #     continue
            if lo['Polar_expression'][1][0] == li:
                tmp.append(lo)
        if len(tmp) > 1:
            first = tmp[0]
            for t in tmp:
                if is_empty(first['Source']) and not is_empty(t['Source']):
                    first['Source'] = t['Source']
                if is_empty(first['Target']) and not is_empty(t['Target']):
                    first['Target'] = t['Target']
        else:
            first = tmp[0]

        new_lst_opinions.append(first)

    return new_lst_opinions


def parse_one_sample(sample):
    raw_text = sample['text']
    rel_id = sample['id']
    lst_opinions = []

    lst_rels = sample['relations']
    lst_ents = sample['entities']

    for rel in lst_rels:
        s, t, e, p = parse_one_opinion(rel, lst_ents, raw_text)
        lst_opinions.append(
            {
                "Source": s,
                "Target": t,
                "Polar_expression": e,
                "Polarity": p,
                "Intensity": "Standard"
            }
        )

    merge_lst_options = merge_expression(lst_opinions)
    return {
        "sent_id": rel_id,
        "text": raw_text,
        "opinions": merge_lst_options
    }


def transform_data(path="all.jsonl", save="formatted_data.json"):
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    full_data = []
    for json_str in json_list:
        result = json.loads(json_str)
        full_data.append(result)

    new_formatted_data = []
    for d in full_data:
        new_formatted_data.append(parse_one_sample(d))

    print(len(new_formatted_data))
    with open(save, 'w') as f:
        json.dump(new_formatted_data, f, indent=4, ensure_ascii=False)


def make_train_dev_test():
    with open('dataset/all.jsonl', 'r') as json_file:
        json_list = list(json_file)

    full_data = []
    for json_str in json_list:
        result = json.loads(json_str)
        full_data.append(result)

    train, dev, test = np.split(np.array(full_data), [int(len(full_data)*0.7), int(len(full_data)*0.8)])

    print(len(train))
    with open('dataset/raw/train.jsonl', 'w') as f:
        for entry in train:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    f.close()

    print(len(dev))
    with open('dataset/raw/dev.jsonl', 'w') as f:
        for entry in dev:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    f.close()

    print(len(test))
    with open('dataset/raw/test.jsonl', 'w') as f:
        for entry in test:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    # with open('dataset/all.jsonl', 'r') as json_file:
    #     json_list = list(json_file)
    #
    # full_data = []
    # for json_str in json_list:
    #     result = json.loads(json_str)
    #     full_data.append(result)

    # new_formatted_data = []
    # for d in full_data:
    #     new_formatted_data.append(parse_one_sample(d))
    #
    # with open('formatted_data3.json', 'w') as f:
    #     json.dump(new_formatted_data, f, indent=4, ensure_ascii=False)
    #
    # print(len(new_formatted_data))

    # make_train_dev_test()
    transform_data("dataset/raw/train.jsonl", "dataset/transformed/train.json")
    transform_data("dataset/raw/dev.jsonl", "dataset/transformed/dev.json")
    transform_data("dataset/raw/test.jsonl", "dataset/transformed/test.json")
    pass
