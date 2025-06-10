import json
import argparse
import pandas as pd
from typing import List, Dict, Any

def generate_rag_verbalization(row: Dict[str, Any], verbalization_level: str) -> Dict[str, Any]:
    """Generate a RAG-optimized verbalization with structured metadata and searchable text."""
    
    # Parse vehicle categories (clean and format)
    vehicle_categories = [
        v.strip().replace("_", " ") 
        for v in row['vehicle_categories']
    ] if isinstance(row['vehicle_categories'], list) else []
    
    # Core metadata (for filtering)
    metadata = {
        **({"census_id": row['cens']} if verbalization_level != "district" and  verbalization_level != "zone" else {}),
        "year": row['year'],
        **({"zone": row['desc_zone']} if verbalization_level != "district" else {}),
        "district": row['district'],
        "has_transport": row['n_stops'] > 0,
        "has_accidents": row['tot_accidents'] > 0,
        "public_vehicle_involvement": row['n_public_vehicles'] > 0,
        "vehicle_categories": vehicle_categories,
        "is_transport_desert": row['avg_distan'] == -1,
    }
    
    # Natural language summary (for semantic search)
    if verbalization_level == 'census':
        header = f"In {row['year']}, Census Area {row['cens']} (Statistical Zone {row['desc_zone']}, District {row['district']})"
    elif verbalization_level == 'zone':
        header = f"In {row['year']}, Statistical Zone {row['desc_zone']} (District {row['district']})"
    elif verbalization_level == 'district':
        header = f"In {row['year']}, District {row['district']}"
    
    text = (
        f"{header} "
        f"had {row['tot']} residents ({row['tot_F']} female, {row['tot_M']} male). "
        f"Transport: {row['n_stops']} stops, {row['n_lines']} lines. "
        f"Accidents: {row['tot_accidents']} involving {', '.join(vehicle_categories)}. "
        f"Key demographics: {row['minors']} minors, {row['seniors']} seniors. "
        f"Foreign population: {row['tot_foreig']}."
    )
    
    return {
        **({"id": f"{row['year']}-{row['cens']}"} if verbalization_level == "census" else {}),# Unique ID for retrieval
        **({"id": f"{row['year']}-{row['zone_stat']}"} if verbalization_level == "zone" else {}),# Unique ID for retrieval
        **({"id": f"{row['year']}-{row['district']}"} if verbalization_level == "district" else {}),# Unique ID for retrieval
        "metadata": metadata,
        "text": text,
        # Optional: Include raw fields for hybrid search
        "raw_data": {k: v for k, v in row.items() if k != 'vehicle_categories'}
    }

def combine_lists(lists: List[List[Any]]) -> List[Any]:
    elements = sum(lists, [])
    filtered = [e for e in elements if e != 0 and e != '0']
    return sorted(set(filtered))

def positive_sum(x: pd.Series) -> float:
    return x[x > 0].sum()

def group_by_verbalization_level(df: pd.DataFrame, verbalization_level: str) -> pd.DataFrame:
    """Remove all the features within a buffer of 500 mt from the census and perform the aggregation of the columns of numerical features"""
    if verbalization_level == "zone" or verbalization_level == "district":
        df = df[[col for col in df.columns if not col.endswith('500')]]
        group_columns = ['year', 'cens', 'zone_stat', 'desc_zone', 'district']
        aggregation_columns = [col for col in df.columns if col not in group_columns]
        aggregation_dictionary = {}
        for col in aggregation_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                aggregation_dictionary[col] = positive_sum
            elif df[col].apply(lambda x: isinstance(x, list)).all():
                aggregation_dictionary[col] = combine_lists
            else:
                aggregation_dictionary[col] = lambda x: sorted(set(x))

    """Remove census ids or statistical zones ids and group data based on zone/district"""
    if verbalization_level == "zone":
        columns_to_remove = ["cens"]
        df = df.drop(columns=columns_to_remove)
        df = df.groupby(["year", "zone_stat", "desc_zone", "district"]).agg(aggregation_dictionary).reset_index()
    elif verbalization_level == "district":
        columns_to_remove = ["cens", "zone_stat", "desc_zone"]
        df = df.drop(columns=columns_to_remove)
        df = df.groupby(["year", "district"]).agg(aggregation_dictionary).reset_index()
    return df

def process_csv(csv_path: str, verbalization_level: str) -> list[Dict[str, Any]]:
    """Process CSV into RAG-ready documents."""
    df = pd.read_csv(csv_path)
    
    # Clean vehicle_categories (handle both strings and lists)
    df['vehicle_categories'] = df['vehicle_categories'].apply(
        lambda x: (
            x.strip("[]").replace("'", "").split(", ") 
            if isinstance(x, str) 
            else []
        )
    )

    df = group_by_verbalization_level(df, verbalization_level)

    if verbalization_level == "zone" or verbalization_level == "district":
        postfix = "-" + verbalization_level
        path = "../data/input/"
        name = "population-and-transport" + postfix + ".csv"
        df.to_csv(path + name, index=False)
    
    return [generate_rag_verbalization(row, verbalization_level) for _, row in df.iterrows()]

def main(args):
    path_input   = "../data/input/"
    path_output  = "../data/output/"
    verbalization_level      = args.verbalization_level # census, zone, district

    verbalizations = process_csv(path_input + "population-and-transport.csv", verbalization_level)
    
    # Save for RAG pipeline
    with open(path_output + f"turin_rag_verbalizations_{verbalization_level}.json", "w") as f:
        json.dump(verbalizations, f, indent=2)
    
    print(f"Generated {len(verbalizations)} RAG-optimized documents.")
    print("Sample document:")
    print(json.dumps(verbalizations[0], indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbalization_level", type=str, required=True)
    args = parser.parse_args()
    main(args)