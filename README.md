This repository represents the implementation part of my master thesis "Predicting Shapley 
values in routing problems with machine learning".

ABSTRACT:
 In routing problems, companies and logistics service providers are not only interested
 in identifying the optimal route with the lowest cost, but also in a fair cost allocation
 among the served customers. A highly regarded method for distributing costs is the
 Shapley value with its unique fairness properties. As the highly intensive computation
 of the exact Shapley values limits the method’s applicability to small instances in routing
 problems, this study presents a novel technique for approximating Shapley values using
 machine learning. The approach is based on the generation of problem-specific features
 to capture the underlying structure of the setting. Therefore, a comprehensive numerical
 study using synthetic instances for the traveling salesman problem and the capacitated
 vehicle routing problem is conducted, including a detailed description of the data gen
 eration process as well as the implementation of the machine learning algorithms. A
 diverse set of models is considered for the experiments in order to determine the most
 suitable ones for each context. The analyses show excellent results by outperforming
 straightforward proxies and state-of-the-art approximation methods from the literature
 for both problem settings, leading to accurate Shapley value predictions for all customers
 within seconds. Additionally, the paper provides economic findings on the primary cost
 factors based on a feature analysis, as well as a study on the computational efficiency
 of the applied models. Importantly, the generalizability of the methodology to further
 operations research contexts is evaluated by applying the technique to a variant of the
 bin packing problem (BPP). The promising results for the BPP show that the approach
 can be effectively applied to other contexts.
 
 Keywords: Shapley value · Cost allocation · Traveling salesman problem · Capacitated
 vehicle routing problem · Machine learning
