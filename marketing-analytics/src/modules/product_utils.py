from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def create_association_rules_including_categories(orders, products, included_categories=None, min_support=0.003, min_confidence=0.75, min_lift=1):
    """
    Create association rules including only specific categories and add the number of transactions in which the rule occurs.

    :param orders: DataFrame containing the orders.
    :param products: DataFrame containing the products.
    :param included_categories: List of categories to include in the analysis. If None, include all categories.
    :param min_support: The minimum support for the Apriori algorithm.
    :param min_lift: The minimum lift for the association rules.
    :return: DataFrame containing the association rules.
    """
    if included_categories is not None:
        # Filter orders to include only the specified categories
        included_products = products[products['product_class'].isin(included_categories)]['product_id']
        filtered_orders = orders[orders['product_id'].isin(included_products)]
    else:
        filtered_orders = orders

    # Filter only purchases (excluding returns)
    purchases = filtered_orders[filtered_orders['direction'] != -1]

    # Group by order and create a list of products purchased in each order
    basket = purchases.groupby('order_id')['product_id'].apply(list).reset_index()

    # Create a matrix of binary indicators for each product in each order
    te = TransactionEncoder()
    te_ary = te.fit(basket['product_id']).transform(basket['product_id'])
    basket_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply the Apriori algorithm to find frequent itemsets with progress bar
    with tqdm(total=len(basket_encoded.columns), desc='Calculating frequent itemsets') as pbar:
        frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True, low_memory=True, verbose=1)
        pbar.update(len(basket_encoded.columns))

    # Create association rules using lift with progress bar
    with tqdm(total=len(frequent_itemsets), desc='Generating association rules') as pbar:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        pbar.update(len(frequent_itemsets))

    rules = rules[rules['lift'] >= min_lift]
    
    # Add the number of transactions in which the rule occurs
    num_transactions = basket_encoded.shape[0]
    rules['num_transactions'] = rules['support'] * num_transactions

    return rules

def get_product_classes(itemsets, products_df):
    product_classes = set()
    for item in itemsets:
        product_class = products_df.loc[products_df['product_id'] == item, 'product_class'].values[0]
        product_classes.add(product_class)
    return tuple(product_classes)

def get_scatter_plot(rules, title):
    # Scatter plot of the rules
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='Reds', alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.colorbar(scatter, label='Lift')
    plt.title(title)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True)
    plt.show()



def plot_association_rules_graph(rules, products, title='Network Graph of Association Rules Colored by Product Class'):
    """
    Plot a network graph of association rules with nodes colored by product class.

    :param rules: DataFrame containing the association rules.
    :param products: DataFrame containing the product information.
    :param title: Title of the plot.
    """
    # Create directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for index, rule in rules.iterrows():
        antecedent = ' '.join([str(item) for item in list(rule['antecedents'])])
        consequent = ' '.join([str(item) for item in list(rule['consequents'])])
        G.add_edge(antecedent, consequent, weight=rule['lift'])

    # Create a list of colors for the product classes
    product_classes = list(range(1, 15))  # Assuming there are 14 product classes
    colors = plt.cm.tab20(range(len(product_classes)))

    # Create a list of colors for the nodes based on product_class
    node_colors = []
    for node in G.nodes():
        product_id = int(node.split(' ')[0])
        node_class = rules[rules['antecedents'].apply(lambda x: product_id in x)]['antecedent_classes'].values
        if len(node_class) == 0:  # Check if the product_id is in consequents
            node_class = rules[rules['consequents'].apply(lambda x: product_id in x)]['consequent_classes'].values

        if len(node_class) > 0:
            node_class = node_class[0][0]
            color_index = product_classes.index(node_class)
            node_colors.append(colors[color_index])
        else:
            node_colors.append('grey')  # Default color if no class found

    # Position nodes in a circular layout
    pos = nx.circular_layout(G)

    # Draw the graph with node colors
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_size=200, node_color=node_colors, arrows=True, font_size=8)



    # Add legend for the product class colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{cls}',
                                  markersize=10, markerfacecolor=colors[idx]) for idx, cls in enumerate(product_classes)]
    plt.legend(handles=legend_elements, title='Product Classes', loc='upper right')

    plt.title(title)
    plt.show()

def analyze_class_associations(rules, products):
    """
    Analyze frequent associations between product classes.

    :param rules: DataFrame containing the association rules.
    :param products: DataFrame containing the products.
    :return: DataFrame containing associations between product classes ordered by frequency.
    """
    # Join the rules with product classes
    rules['antecedent_classes'] = rules['antecedents'].apply(lambda x: get_product_classes(x, products))
    rules['consequent_classes'] = rules['consequents'].apply(lambda x: get_product_classes(x, products))

    # Analyze the frequency of associations between product classes
    class_associations = rules.groupby(['antecedent_classes', 'consequent_classes']).size().reset_index(name='frequency')
    class_associations = class_associations.sort_values(by='frequency', ascending=False)

    return class_associations
