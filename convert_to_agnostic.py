import json

def modify_results_categories(my_results_path,res):
    with open(my_results_path, 'r') as f:
        my_results_data = json.load(f)

    if 'categories' in my_results_data:
        for category in my_results_data['categories']:
            category['id'] = 1

    if 'annotations' in my_results_data:
        for annotation in my_results_data['annotations']:
            annotation['category_id'] = 1

    with open(res, 'w') as f:
        json.dump(my_results_data, f, indent=2)


my_results_path = './Final_test/my_results.json'
res='./Final_test/my_results_agnostic.json'
modify_results_categories(my_results_path,res)
