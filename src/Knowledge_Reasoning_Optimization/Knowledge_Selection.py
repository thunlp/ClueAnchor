import json
import argparse
from collections import defaultdict

def process_file(input_file, output_file, target_data_types):
    stats = defaultdict(int)

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            data_type = record.get("data_type")

            if data_type not in target_data_types:
                continue
            
            i_score = record.get("internal_score", 0.0)
            e_score = record.get("external_score", 0.0)
            c_score = record.get("clueanchor_score", 0.0)
            v_score = record.get("verify_score", 0.0)

            has_internal_think = bool(record.get("internal_think"))
            has_external_think = bool(record.get("external_think"))
            has_clueanchor_think = bool(record.get("clueanchor_think"))

            if i_score == e_score == c_score == 1.0 and has_internal_think:
                stats["case0"] += 1
                continue

            def write_case(chosen_think, chosen_answer, reject_think, reject_answer, case_name):
                new_record = {
                    "passages": record["passages"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "data_type": data_type,
                    "chosen_think": chosen_think,
                    "chosen_answer": chosen_answer,
                    "reject_think": reject_think,
                    "reject_answer": reject_answer,
                }
                outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                stats[case_name] += 1

            if i_score == 1.0 and has_internal_think:
                if e_score == 1.0 and c_score == 0.0:
                    write_case(record["internal_think"], record["internal_answer"], 
                               record["clueanchor_think"], record["clueanchor_answer"], "case1")
                elif e_score == 0.0 and c_score == 1.0:
                    write_case(record["internal_think"], record["internal_answer"], 
                               record["external_think"], record["external_answer"], "case2")
                elif e_score == 0.0 and c_score == 0.0:
                    write_case(record["internal_think"], record["internal_answer"], 
                               record["external_think"], record["external_answer"], "case3")
            else:
                if e_score == 1.0 and c_score == 1.0 and has_external_think:
                    write_case(record["external_think"], record["external_answer"], 
                               record["internal_think"], record["internal_answer"], "case4")
                elif e_score == 1.0 and c_score == 0.0 and has_external_think:
                    write_case(record["external_think"], record["external_answer"], 
                               record["internal_think"], record["internal_answer"], "case5")
                elif e_score == 0.0 and c_score == 1.0 and v_score ==1.0 and has_clueanchor_think:
                    write_case(record["clueanchor_think"], record["clueanchor_answer"], 
                               record["internal_think"], record["internal_answer"], "case6")
                elif e_score == 0.0 and c_score == 0.0:
                    stats["case7"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Process RAG training data and extract useful records.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
    parser.add_argument("--target_data_types", nargs="+", default=["hotpotqa", "NQ", "SQuAD", "TriviaQA", "2wikiMQA"],
                        help="List of data types to include")
    args = parser.parse_args()
    stats = process_file(args.input_file, args.output_file, args.target_data_types)

    print(f"\nSelecting completed. Results saved to {args.output_file}")
    for i in range(8):
        print(f"Case {i}: {stats.get(f'case{i}', 0)} records")
        

if __name__ == "__main__":
    main()
