# **Unsupervised discovery of Interpretable Visual Concepts**

## Authors: Caroline Mazini Rodrigues, Nicolas Boutry, and Laurent Najman

Providing interpretability of deep-learning models to non-experts, while fundamental for a responsible real-world usage, is challenging. Attribution maps from xAI techniques, such as Integrated Gradients, are a typical example of a visualization technique containing a high level of information, but with difficult interpretation. In this paper, we propose two methods, Maximum Activation Groups Extraction (MAGE) and Multiscale Interpretable Visualization (Ms-IV), to explain the model's decision, enhancing global interpretability. MAGE finds, for a given CNN, combinations of features which, globally, form a semantic meaning, that we call concepts. We group these similar feature patterns by clustering in “concepts”, that we visualize through Ms-IV. This last method is inspired by Occlusion and Sensitivity analysis (incorporating causality), and uses a novel metric, called Class-aware Order Correlation (CaOC), to globally evaluate the most important image regions according to the model's decision space. We compare our approach to xAI methods such as LIME and Integrated Gradients. Experimental results evince the Ms-IV higher localization and faithfulness values. Finally, qualitative evaluation of combined MAGE and Ms-IV demonstrate humans' ability to agree, based on the visualization, on the decision of clusters' concepts; and, to detect, among a given set of networks, the existence of bias.

---

The code was developed in Python (64-bit) 3.8.8, Numpy 1.18.5, and Pytorch 1.12.0. 

List of used libraries:

- matplotlib (3.2.2)
- torchvision (0.13.0)
- Pillow (8.1.0)
- umap (0.1.1)
- sklearn (1.1.2)
- scipy (1.4.1)
- pandas (1.2.1)
- tqdm (4.64.0)
- kneed (0.8.1)

---

The method involves two main steps:

1. Concepts decomposition with Maximum Activation Groups Extraction (MAGE);
2. Concepts visualization with Multiscale Interpretable Visualization (Ms-IV);

We also provide the training scripts used for both network architectures and datasets.

## Supporting scripts:

- causal_viz/: scripts to obtain feature maps' representation and auxiliate visualizations
	- images_by_patches.py: auxiliary functions to manipulate patches by coordenates
	- occlude_images_model.py: occluded image predictions to analyze impact (causality from the occlusion)
	- representation_maps.py: functions to generate feature maps' representation and to visualize them
- utils_caoc/: scripts to calculate rankings and ranking metrics
	- order_functions.py: create rankings
	- kendall_tau_correlations.py: calculates caoc and other metrics
- utils_model/: scripts to help to manipulate the models
	- crop_feat_maps.py: to predict cropped image parts
	- test_individual_feat_maps.py: to predict images with model's occluded feature maps

## Output folders:

- checkpoints/: files of network weights;
- viz_images/ folder with the resultant visualizations from ms_iv.py script;
- save_clusters/: files from mage.py containing:
    - images’ activations;
    - patch’s parameters;
    - feature maps’ representation;
    - the clusters’ evaluation;
    - final clusters labels, and;
    - indexes from more important images per cluster.

## Datasets folder:

datasets/: folder in which the datasets should be placed. The experiments were performed for two datasets:

- cat_dog:
        
  Download from: [https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data)
        
- CUB:
  
  Download from: [https://data.caltech.edu/records/65de6-vp158](https://data.caltech.edu/records/65de6-vp158)
  
  Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2022). CUB-200-2011 (1.0) [Data set]. CaltechDATA. [https://doi.org/10.22002/D1.20098](https://doi.org/10.22002/D1.20098)

  X. He and Y. Peng, "Fine-Grained Visual-Textual Representation Learning," in *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 30, no. 2, pp. 520-531, Feb. 2020, doi: 10.1109/TCSVT.2019.2892802.

After downloading the datasets, include the images inside the folder images/ from the corresponding dataset.

**Example:**
```
datasets/cat_dog/images/cat.1.jpg
datasets/CUB/images/158.Bay_breasted_Warbler/Bay_Breasted_Warbler_0026_159744.jpg
```        

We will detail the content of the two main steps.

## 1) mage:

The extraction of features uses three different networks

**Example**

```
python mage.py --dataset cat_dog --gpu_ids 0 --model vgg
```

## 2) ms_iv:

**Example**

With concepts visualization

```
python concepts_ms_iv.py --dataset cat_dog --gpu_id 0 --model vgg
```

With network visualization (without concepts decomposition) for one image

```
python net_ms_iv.py --dataset cat_dog --gpu_id 0 --model vgg --idx_img 0
```

## For training the models

- training_scripts/train.py: used to train the four analyzed models.

**************Example**************

Cat_dog dataset:

```
python train.py --dataset cat_dog --gpu_id 0 --model vgg
```

CUB dataset:

```
python train.py --dataset CUB --gpu_id 0 --model vgg
```

## Citation

The presented code is the implementation of the paper entitled **Unsupervised discovery of Interpretable Visual Concepts**. If you find it useful in your research, please cite our paper:

```
@article{rodrigues2023unsupervised,
      title={Unsupervised discovery of Interpretable Visual Concepts}, 
      author={Caroline Mazini Rodrigues and Nicolas Boutry and Laurent Najman},
      year={2023},
      eprint={2309.00018},
      journal={arXiv},
      doi={10.48550/arXiv.2309.00018}
}
```
