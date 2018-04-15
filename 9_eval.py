def _load_data(submit_file, reference_file):
    # load submit result and reference result

    with open(submit_file, 'r') as f:
        submit_data = f.readlines()

    with open(reference_file, 'r') as f:
        ref_data = f.readlines()

    submit_dict = {}
    ref_dict = {}

    for each_line in submit_data:
        item = each_line.split()
        submit_dict[item[0]] = item[1]

    for each_line in ref_data:
        item = each_line.split()
        ref_dict[item[0]] = item[1]

    return submit_dict, ref_dict


def _eval_result(submit_file, reference_file):
    # eval the error rate

    result = {
        'err_code': 0,
        'error': '0',
        'warning': '0',
        'score': '0'
    }

    try:
        submit_dict, ref_dict = _load_data(submit_file, reference_file)
    except Exception as e:
        result['err_code'] = 1
        result['error'] = str(e)
        return result

    right_count = 0

    keys = tuple(submit_dict.keys())
    for (key, value) in ref_dict.items():
        if key not in keys:
            result['warning'] = 'lacking image in your submitted result'
            print('warning: lacking image %s in your submitted result' % key)
            continue
        if submit_dict[key] == value:
            right_count += 1

    result['score'] = str(float(right_count) / len(ref_dict))

    return result


result = _eval_result(f'pred_{superclass}.txt',
                      f'ans_{animals_fruits}_true.txt')
print(result)
