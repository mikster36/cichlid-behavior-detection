# cichlid-behavior-detection
a machine learning model to classify behavior in East African cichlid fish.<br />

![velocity](https://github.com/mikster36/cichlid-behavior-detection/assets/74252522/fe8d77a6-bc98-463a-ad6e-01ff9d9b7527)

## about
the McGrath lab is focused on studying the evolution of behavior and uses Lake Malawi Cichlids as model organisms as they have rapidly speciated yet remain genetically similar. this code is part of a larger project focused on mapping genes, neurons, and cell types to particular behaviors. our code helps add valuable data regarding the counts and types of behaviors displayed in different trials.<br />
## approach
previously, the automated behavior detection system has used hidden markov chains and [convolutional residual networks](https://github.com/ptmcgrat/CichlidActionClassification) to detect behaviors (feeding, bower construction, and quivering) through sand change and [depth sensing](https://github.com/tlancaster6/CichlidBowerTracker) to classify bower shapes (pit vs castle). we build upon this system by using pose estimation supported by [DeeplabCut](https://github.com/DeepLabCut/DeepLabCut) to identify behaviors undetectable by sand change alone. these include mating and aggressive behaviors like bower-circling/spawning, display, pursue, chase/flee, and bite. other work in the lab surrounding this project includes sex detection, which may be integrated into this code to assist with identifying behaviors.
## data
our dataset (collected in 2020) consists of ~10 hour top-down videos of 38 tanks of 4 female, 1 male Mchenga conophoros. 19 of these trials are behavioral trials (a bower is built during the video collection period), and the remaining 19 are control trials (no bower is build during this period).
## pipeline
![Behavior Classification Workflow](https://github.com/mikster36/cichlid-behavior-detection/assets/74252522/b8aa2456-281d-491d-afc7-6f9a2d66eaa4)


## references
Johnson, Z.V., Arrojwala, M.T.S., Aljapur, V. et al. Automated measurement of long-term bower behaviors in Lake Malawi cichlids using depth sensing and action recognition. Sci Rep 10, 20573 (2020).<br /> https://doi.org/10.1038/s41598-020-77549-2.<br /><br />
Lijiang Long, Zachary V. Johnson, Junyu Li, Tucker J. Lancaster, Vineeth Aljapur, Jeffrey T. Streelman, Patrick T. McGrath. Automatic Classification of Cichlid Behaviors Using 3D Convolutional Residual Networks. iScience Volume 23, Issue 10, 2020.<br />https://doi.org/10.1016/j.isci.2020.101591.<br /><br />
Mathis, A., Mamidanna, P., Cury, K.M. et al. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci 21, 1281–1289 (2018).
<br />https://doi.org/10.1038/s41593-018-0209-y.<br /><br />
York, Ryan & Patil, Chinar & Hulsey, C. & Streelman, J. & Fernald, Russell. (2015). Evolution of bower building in Lake Malawi cichlid fish: Phylogeny, morphology, and behavior. Frontiers in Ecology and Evolution. 3.<br />https://doi.org/10.3389/fevo.2015.00018.<br />
