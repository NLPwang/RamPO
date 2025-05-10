def extract_gr_and_pre(log):
    gr_values = []
    pre_values = []

    for line in log:
        line = line.strip()
        if '====GR====' in line and '====Pre====' in line:
            parts = line.split('====')
            gr_value = parts[2].lower().strip().replace('.','').replace('[\'',"").replace("\']","")
            pre_value = parts[4].strip().lower().strip().replace('.','')
            gr_values.append(gr_value)
            pre_values.append(pre_value)

    return gr_values, pre_values


def calculate_accuracy(gr_values, pre_values):
    correct = 0
    total = len(gr_values)
    for gr, pre in zip(gr_values, pre_values):
        if gr in pre:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

def evaluate(log):
    gr_values, pre_values = extract_gr_and_pre(log)
    accuracy = calculate_accuracy(gr_values, pre_values)
    print(f"acc: {accuracy:.2%}")
