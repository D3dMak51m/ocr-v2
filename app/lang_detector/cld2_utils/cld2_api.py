import pycld2 as cld2
import pandas as pd


def detect_lang_cld2(text):
    result_arr = []

    try:
        isReliable, textBytesFound, details, vectors = cld2.detect(
            text, returnVectors=True
        )

        details_arr, lang_set = parse_detail(details)

        final_arr = handle_vectors_via_pandas(
            vectors, details_arr, lang_set, textBytesFound
        )
        result_arr = final_arr
    except Exception as e:
        print(f"Error: {e}", flush=True)
    return result_arr


def parse_detail(details):
    lang_set = set()
    result_arr = []
    # details is a list of tuples:
    # example: ('ENGLISH', 'en', 25, 825.0)
    lang_set.add("un")
    for detail in details:
        lang_name = detail[1]
        if lang_name == "un":
            continue
        lang_set.add(lang_name)
        score = round((detail[2] / 100), 4)
        # result_arr.append({'lang': lang_name, 'prob': score})
        result_arr.append({"language": lang_name, "score": score})

    return result_arr, lang_set


def handle_vectors_via_pandas(
    vectors, lang_list_details, lang_set_to_exclude, textBytesFound
):
    df_vectors = pd.DataFrame(
        list(vectors), columns=["lang_offset", "lang_len", "lang_name", "language"]
    )
    df_vectors = df_vectors[~df_vectors.language.isin(lang_set_to_exclude)]
    df_vectors.drop(["lang_name"], axis=1, inplace=True)

    df_vectors = df_vectors.groupby("language").sum().reset_index()
    col_prob = df_vectors.apply(lambda row: row.lang_len / textBytesFound, axis=1)
    df_vectors["score"] = round(col_prob, 4)

    df_vectors.drop(["lang_len", "lang_offset"], axis=1, inplace=True)

    df_vectors = pd.concat([df_vectors, pd.DataFrame.from_records(lang_list_details)])

    df_vectors.sort_values(by=["score"], ascending=False, inplace=True)
    return df_vectors.to_dict("records")
