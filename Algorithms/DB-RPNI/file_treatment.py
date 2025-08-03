import json


def parse_input_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    if not lines:
        raise ValueError("The file is empty.")

    # 第1行: 合法符号
    all_symbols = lines[0].split(" ")
    if not all_symbols:
        raise ValueError("The first line is wrong.")

    # 从第2行起，每行一个trace
    traces = []
    for idx, line in enumerate(lines[1:], start=2):
        if "->" not in line:
            raise ValueError(f"The {idx} line lack of '->': {line}")

        parts = line.split("->")
        if len(parts) != 2:
            raise ValueError(f"The {idx} line contains many '->': {line}")

        inputs_str = parts[0].strip()
        outputs_str = parts[1].strip()

        if not (outputs_str.startswith("[") and outputs_str.endswith("]")):
            raise ValueError(f"The {idx} line's output should be in [ ]: {outputs_str}")

        outputs_str_inner = outputs_str[1:-1].strip()
        if not outputs_str_inner:
            raise ValueError(f"The {idx} line's output is empty: {line}")

        output_symbols = [x.strip() for x in outputs_str_inner.split(", ")]
        input_symbols = inputs_str.split(" ")
        if len(input_symbols) != len(output_symbols):
            raise ValueError(
                f"The {idx} line's input length is ({len(input_symbols)}) while its output length is ({len(output_symbols)})!"
            )

        trace_pairs = list(zip(input_symbols, output_symbols))
        traces.append(trace_pairs)

    return all_symbols, traces


def assign_state_ids_and_dump_json(root, out_file: str, reachable) -> None:
    node_list = list(reachable)

    # 给节点分配 q0, q1, ...
    node2id = {}
    for i, nd in enumerate(node_list):
        node2id[nd] = f"q{i}"

    machine_dict = {
        "initial_state": node2id[root],
        "states": {},
    }

    for nd in node_list:
        st_id = node2id[nd]
        fp = dict(nd.fingerprint)
        tr = {}
        for sym, child in nd.transitions.items():
            if child in node2id:  # 只考虑可达child
                tr[sym] = node2id[child]
        machine_dict["states"][st_id] = {
            "fingerprint": fp,
            "transitions": tr
        }

    machine_dict["state_count"] = len(node_list)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(machine_dict, f, indent=2, ensure_ascii=False)
    print(f"Mealy machine file: {out_file}, state number={len(node_list)}")
