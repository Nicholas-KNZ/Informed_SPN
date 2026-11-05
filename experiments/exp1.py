'''
Vergleiche für 5 verschiedene Seeds die folgeneden Modellkonfigurationen: 
    
    - LearnSPN (nur einmalig pro Seed ) 
    
    1 - Modellgröße (Repetitions = 1 - Region Nodes = 4 - Output Nodes = 4)
    
        - RAT SPN 
        - IST SPN Heuristic + G-Test 
        - IST SPN Heuristic + RCA 
        - IST SPN Optimal + G-Test 
        - IST SPN Optimal + RCA 
        - IST SPN Random + G-Test 
        - IST SPN Random + RCA

'''

from utils import set_seed, get_cancer_data, print_model_statistics
from spflow.learn import learn_spn, gradient_descent, expectation_maximization
from spflow.meta import Scope
from spflow.modules.leaves.normal import Normal
from spflow.zoo.rat.rat_spn import RatSPN
from spflow.utils.visualization import visualize
from src.ist import InformedSplitTensorizedSPN
from src import ist
import time
import torch 


def main(): 

    models = []
    dataset = get_cancer_data()[0]

    for seed in range(5):

        print(f"[SEED {seed}]")

        

        # Learn SPN 
        '''
        set_seed(seed) 
        scope = Scope(list(range(0, dataset.num_feature)))
        lsp_leaves = Normal(scope=scope)
        start_time = time.time()
        learned_spn = learn_spn(dataset.train_tensor, lsp_leaves)
        train_time = time.time() - start_time
        expectation_maximization(learned_spn, dataset.train_tensor)
        train_likelihood = torch.mean(learned_spn.log_likelihood(dataset.train_tensor)).item()
        test_likelihood = torch.mean(learned_spn.log_likelihood(dataset.test_tensor)).item()
        models.append({"seed": seed, "dataset": dataset.name, "model": "Learn-SPN", "structure_learned [t]": train_time, "train_likelihood": train_likelihood, "test_likelihood": test_likelihood})
        print(f"Learn SPN")
        print(f"Train {train_likelihood}")
        print(f"Test {test_likelihood}")
        '''

        # RAT SPN 
        set_seed(seed) 

  
        rat_leaves = [Normal(scope=i, out_channels=4, num_repetitions=1) for i in range(dataset.num_feature)]

        start_time = time.time()
        rat = RatSPN(leaf_modules=rat_leaves, n_root_nodes=1, n_region_nodes=8, num_repetitions=1, depth = 3)
        train_time = time.time() - start_time

        # expectation_maximization(rat, dataset.train_tensor)
        gradient_descent.train_gradient_descent(
                        rat,
                        dataset.train_dataloader,
                        epochs=100,
                    )
        train_likelihood = torch.mean(rat.log_likelihood(dataset.train_tensor)).item()
        test_likelihood = torch.mean(rat.log_likelihood(dataset.test_tensor)).item()

        models.append({"seed": seed, "dataset": dataset.name, "model": "RAT-SPN", "structure_learned [t]": train_time, "train_likelihood": train_likelihood, "test_likelihood": test_likelihood})

        print(f"RAT SPN")
        print(f"Train {train_likelihood}")
        print(f"Test {test_likelihood}")

        # IST SPN 
        versions = [{'sorting': 'heuristic', 'independence_test': 'gtest'},
                    {'sorting': 'heuristic', 'independence_test': 'rdc'},
                    {'sorting': 'optimal', 'independence_test': 'gtest'},
                    {'sorting': 'optimal', 'independence_test': 'rdc'},
                    {'sorting': 'random', 'independence_test': 'gtest'},]

        for version in versions: 
            set_seed(seed) 

            ordering = version['sorting']
            test = version['independence_test']
            tree_structure = InformedSplitTensorizedSPN(data = dataset.train_tensor,
                           root_sum_node = 1,
                           n_region_nodes = 8,
                           num_repetitions = 1,
                           depth = 3,
                           dependency_test=test, 
                           ordering=ordering)

            # expectation_maximization(tree_structure.root_node, dataset.train_tensor)
            gradient_descent.train_gradient_descent(
                tree_structure.root_node,
                dataset.train_dataloader,
                epochs=100,
            )


            train_likelihood = torch.mean(tree_structure.root_node.log_likelihood(dataset.train_tensor)).item()
            test_likelihood = torch.mean(tree_structure.root_node.log_likelihood(dataset.test_tensor)).item()

            models.append({"seed": seed, "dataset": dataset.name, "model": f"IST_{ordering}_{test}", "structure_learned [t]": train_time, "train_likelihood": train_likelihood, "test_likelihood": test_likelihood})

            print(f"IST_{ordering}_{test}")
            print(tree_structure.get_tree())
            print(f"Train {train_likelihood}")
            print(f"Test {test_likelihood}")

    print("Final Report")
    print_model_statistics(models)

            
if __name__ == '__main__':
    main()

               
    

  