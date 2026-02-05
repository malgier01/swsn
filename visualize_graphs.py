import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_csv_row(csv_path, row_index=0, output_dir="output_graphs"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    if row_index >= len(df):
        print(f"Error: Row index {row_index} out of bounds. Max is {len(df)-1}.")
        return

    row = df.iloc[row_index]
    answer_id = row.get('id', 'Unknown')
    print(f"Visualizing Graph for Answer ID: {answer_id}")

    G = nx.DiGraph()

    color_map = {
        "Accuracy": "#FF9999",     # Red
        "Completeness": "#99CCFF", # Blue
        "Empathy": "#99FF99",      # Green
        "Input": "#FFFF99"         # Yellow
    }

    criteria_list = ["Accuracy", "Completeness", "Empathy"]

    for criteria in criteria_list:
        col_name = f"{criteria}_graph"

        if col_name in row and pd.notna(row[col_name]):
            try:
                graph_data = json.loads(row[col_name])
                edges = graph_data.get('edges', [])
                if not edges: continue

                for edge in edges:
                    source = edge.get('source')
                    target = edge.get('target')
                    relation = edge.get('relationship', 'relates')

                    if source and target:
                        G.add_edge(source, target, label=relation, agent=criteria)

                        if source not in G.nodes:
                            G.nodes[source]['agent'] = criteria
                        if target not in G.nodes:
                            G.nodes[target]['agent'] = criteria

            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON for {criteria} in row {row_index}")

    if G.number_of_nodes() == 0:
        print(f"Graph is empty for ID {answer_id}. No edges found.")
        return

    pos = nx.spring_layout(G, k=0.8, iterations=50)
    plt.figure(figsize=(14, 10))

    colors = [color_map.get(G.nodes[n].get('agent', 'Input'), '#CCCCCC') for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=colors, alpha=0.9, edgecolors='black')

    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif",
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray', arrowsize=25, min_source_margin=20, min_target_margin=20)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    plt.title(f"Critique Graph for Answer #{answer_id}")
    plt.axis('off')
    plt.tight_layout()
    
    # Zapisz do pliku zamiast wyświetlać
    filename = os.path.join(output_dir, f"graph_id_{answer_id}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved graph to {filename}")


if __name__ == "__main__":
    csv_file_path = 'FINAL_threaded_report_with_graphs.csv'
    output_dir = "output_graphs"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(csv_file_path):
        print(f"Plik {csv_file_path} nie istnieje. Uruchom najpierw skrypt oceniający.")
    else:
        # Możesz zmienić zakres, aby wygenerować mniej wykresów, np. range(0, 5)
        df = pd.read_csv(csv_file_path)
        total_rows = len(df)
        print(f"Znaleziono {total_rows} wierszy do przetworzenia.")
        
        for i in range(total_rows):
            try:
                visualize_csv_row(csv_file_path, row_index=i, output_dir=output_dir)
            except Exception as e:
                print(f"Error in row {i}: {e}")