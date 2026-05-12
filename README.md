# Machine-Unlearning-In-Applied-Contexts

## Introduction: 
Generative face models do not just learn what faces look like in general, they learn specific people. A face autoencoder trained on celebrity images can reconstruct any of those people from a noisy or partial input. That is what makes them useful, and also what makes them ethically and legally complicated. Once a person’s face has been absorbed into the model’s weights, the obvious question is whether they can be removed without retraining the model from scratch. The technical name for this problem is machine unlearning. 
Our project explores unlearning at a small scale on face autoencoders, working with two kinds of targets. The first is a binary visual attribute (glasses) that sits on top of a face without changing the rest of it. This is a clean, well-defined feature that gives us a controlled testbed for our methods. The second is a single person’s identity, which is much harder. Identity is not one isolated feature but a distributed combination of face shape, skin tone, eye spacing, hair, expression, and even the lighting and background patterns that tend to appear in that person’s photos. 
The central question is whether techniques that work for a clean binary attribute like glasses still work when the thing we want the model to forget is a specific person. 

## Problem statement: 
The motivation is privacy. Two of the largest face datasets ever assembled, VGGFace2 (Oxford, 9,131 identities, 3.31M images) and MS-Celeb-1M (Microsoft, ~10M images of ~100K “celebrities”) have been pulled by their creators. MS-Celeb-1M was removed in 2019 after journalists revealed that many of the so-called celebrities were private individuals who had not consented. Oxford has since removed VGGFace2 without public explanation.
The deeper problem is that pulling a dataset doesn’t pull the models trained on it. They still encode every person they saw. The General Data Protection Regulation (GDPR) Article 17 grants a “right to erasure” of personal data, and scholars have argued this should extend to models trained on that data. But retraining a foundation model from scratch every time someone requests removal isn’t realistic because modern diffusion models cost millions of dollars and weeks of computation to train. 
Machine unlearning, the research field that addresses this gap, develops methods for surgically removing specific training data influence from a trained model. For generative models, identity removal is one of the major cases with methods like Erased Stable Diffusion and Unified Concept Editing. Our project tests a simpler family of techniques on a much smaller scale: gradient ascent, random noise targeting, good-teacher/bad-teacher distillation, and latent vector arithmetic. We apply them first to a binary attribute (glasses) and then to a single celebrity’s identity. 

## Research Questions: 
How do the four methods compare on the same forget/retain metrics? Which produces the cleanest tradeoff between forgetting the target and preserving everything else?
How does the difficulty change between the two targets, and what does that tell us about how the identity is represented in the latent space?
Does latent vector arithmetic, the technique that worked for binary attributes in early generative-model work, transfer to identity?

## Data
For unlearning glasses problem: CelebA Dataset, available here: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and on Kaggle here: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

Unlearning celebrities problem: Celebrity Face Image Dataset from Kaggle found here: https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset/data

## Methodology & Data
### Theory
The methods in this project depend on two basic properties of generative models. We expect to use AutoEncoders and Variational AutoEncodes. To carry out this project, we will require sufficient data to train and tune our models. We expect to download facial data from Kaggle, where we can utilize more data than would be possible by ourselves. The methods we used were:

#### Gradient Ascent
This method unlearns the “forget set” by training the model to ascend towards the loss gradient instead of descending away from it. This reverses the natural gradient descent learning, thereby unlearning the forget set’s data. Further into the project, we implemented retain steps, which reinforced classical machine learning with gradient descent steps on the retain set data.

#### Random Noise Targeting
This method is similar to gradient ascent, however the model is trained to reconstruct random noise instead of the direct opposite of the forget set images. This has the effect of lowering the retain set loss, because since the retain set and forget set often share features, gradient ascent causes large retain set losses. This method attempts to mitigate unlearning the features we desire to retain

#### Good Teacher / Bad Teacher: 
This method initializes a good teacher (the original trained model with its weights frozen), a bad teacher (can be many different things, usually either model with random weights or random shuffled samples from the dataset), and a student (the good teacher, however this is the model we train). During training, on images in the retain set, we use normal gradient descent on the good teacher’s reconstructions (however when we have access to the data, this can be replaced with normal training towards the images themselves). On images in the forget set, we train towards the output of the bad teacher. This method maximizes the loss on the forget set while minimizing the loss on the retain set.

### AutoEncoders
Autoencoders compress an image into a low-dimensional latent code and reconstruct the image from that code; during training, the latent space often organizes itself in meaningful ways. The organization can be roughly linear for binary attributes. For example, averaging the latent codes for “men with glasses” and subtracting the average for “men without glasses” produces a direction that, added to another face, makes it appear to have glasses. 

### Variational AutoEncoders
A variational autoencoder adds a term to the training objective that pushes the latent space to be smoother and more continuous, which makes direction arithmetic more reliable but introduces its own failure modes. Because the latent space is smooth and relative, random points when decoded become new unique images, allowing VAEs to generate their own data.

### Latent Vectors
Latent vectors are vectors in the encoded latent space of a particular model. In AutoEncoders, this latent space has holes because a smooth, even area is not enforced. By contrast, VAEs are pushed to enforce this relational space where, like in the input space, similar vectors in the encoded space represent similarly looking images in the output space.

## Methods and Results
### The Beginning
To begin this project, we custom-trained an AutoEncoder on a dataset of 15,000 celebrity images with a plan to use gradient ascent to unlearn “glasses”. We chose glasses initially because they represent a large portion of each dataset, so they demonstrated a ‘proof of concept’ for learning a specific feature, which we would later implement on individual faces. The goal was to unlearn the concept of glasses and then move on to a specific face afterward. However, gradient ascent applied on this AutoEncoder (AE for future reference) was harder to implement than expected. If the steps were too big, the AE would unlearn everything and the test error would explode into fragmented messes of reconstructed images. Too small, and the model could still recreate the glasses, theoretically making the process moot due to privacy concerns. Gradient ascent worked in some aspects, however we needed more methods.

In Figure 1, we have our initial reconstructions using our AutoEncoder (2,432,835 params) trained from scratch on our celebrity dataset, with an initial MSE loss of 0.00523.

##### Figure 1 — AE Reconstructions
![AE Reconstructions](blog_figures/AE_Reconstructions.png)
Generated in [GlassesAutoEncoder.ipynb, Cell 5](https://colab.research.google.com/drive/1QCjcidtB0i2CBfFHQbSnCGguAjw1Yu4J#scrollTo=3zBDP_IEDp4U).

The retain and forget loss are nearly identical, as the model has not yet unlearned the images of glasses.

Next, we will apply Gradient Ascent on the AutoEncoder and measure its retain and forget set losses.

##### Figure 2 — GA Glasses Faces
![AE GA Glasses](blog_figures/AE_GA_Glasses.png)
Generated in [GlassesAutoEncoder.ipynb, Cell 6](https://colab.research.google.com/drive/1QCjcidtB0i2CBfFHQbSnCGguAjw1Yu4J#scrollTo=DRbpA9NgL4CI).

The losses of the GA (Gradient Ascent) unlearning model clearly indicate unlearning the glasses, however they degrade the eye area as a whole as well. 

##### Figure 3 — GA No-Glasses Faces
![AE GA No-Glasses](blog_figures/AE_GA_No-Glasses.png)
Generated in [GlassesAutoEncoder.ipynb, Cell 6](https://colab.research.google.com/drive/1QCjcidtB0i2CBfFHQbSnCGguAjw1Yu4J#scrollTo=DRbpA9NgL4CI).

In Figure 3 are the GA unlearned reconstructions of the faces without glasses, used to show that the model did not simply unlearn the entire dataset. Here, the loss is lower and the faces indicate some level of eye-area deconstruction, however many faces do not and it is far less severe.

* Forget loss increased by: 1082.1%
* Retain loss increased by: 230.4%

### New Methods of AutoEncoder Unlearning
Next, after some research, we applied new methods of unlearning to reduce the MSE loss of the retain set and increase the loss on the forget set. We applied Random Noise Targeting, a method that pushes the forget set images in the model towards randomly generated noise instead of ascending the curve, which should benefit the MSE loss of the retain set by not directly inverting the weights. Next we implemented the Good Teacher / Bad Teacher method. This technique works by instantiating the ‘Good Teacher’ and ‘Bad Teacher’ models (the original, trained model and a model with random weights respectively), as well as a student model (a malleable/trainable copy of the initially trained model). In the training phase, the student’s weights are trained towards the Good Teacher’s reconstructions on the retain set images while training towards the “random” generations of the Bad Teacher on the forget set. The bad teacher, however, is different from the Random Noise Targeting (RNT) in that its output is deterministic, given that the weights in the Bad Teacher are frozen when created (output = bad_teacher_decoder(latents)). 

Below (Figure 4) are the results of the random noise targeting approach to machine unlearning  

##### Figure 4 — Random Noise Reconstructions
![AE Random Noise](blog_figures/AE_Random-Noise.png)
Generated in [GlassesAutoEncoder.ipynb, Cell 7](https://colab.research.google.com/drive/1QCjcidtB0i2CBfFHQbSnCGguAjw1Yu4J#scrollTo=BXS-e7tiEpPF).

* Forget loss rose by: 899.2%
* Retain loss rose by: 82.6%

Finally, below (Figure 5) is the result of the Good Teacher / Bad Teacher unlearning method.

##### Figure 5 — Teacher Reconstructions
![AE Teacher](blog_figures/AE_Teacher.png)
Generated in [GlassesAutoEncoder.ipynb, Cell 8](https://colab.research.google.com/drive/1QCjcidtB0i2CBfFHQbSnCGguAjw1Yu4J#scrollTo=nV2TgKUMFgYx).

* Forget loss rose by: 124.3%
* Retain loss rose by: 37.3%

### Variational AutoEncoders
After working with AE reconstruction, we thought that Variational AutoEncoders (known as VAEs henceforth), which can generate their own images, would aid our research into privacy compliance. Initially, we tried training our own VAE on the celebrity dataset (15,000 images), however this set was far too small. The images created were messy and did not resemble people enough to reasonably violate compliance laws. Instead, we downloaded the stabilityai/sd-vae-ft-mse pretrained VAE, which produced far better reconstructions and generations. We then tested the pVAE’s reconstructions, which look very accurately human-like, more than enough to violate privacy compliance laws if a person didn’t consent to their participation in a VAE reconstruction dataset. Next, we experimented with pVAE latent vector injection.

### Latent Vector Injection
Latent vectors are a feature defined initially in Variational AutoEncoders, constructing a relational multi-dimensional space from the encoded inputs. This means that ‘euclidean’ direction changes in the encoded latent vectors produce images that look similar (as opposed to AutoEncoders, which are not as reliable to produce latent spaces, however they may emerge with enough training). 
#### Hypothesis
with latent vector injection, we may be able to produce modified reconstructions that have specific traits that we extract. We obtained the latent vector direction of glasses by subtracting the mean of the latent vectors of the “no glasses” images from the mean of the latent vectors of the “glasses” images. This gave us the mean difference between the encoded images in the latent space that had glasses and those that did not, producing a “direction” that would turn non-glasses images into glasses images.

### Latent Vector Experimentation
After obtaining the glasses latent direction vector, our next step was to test it on images. First, we tested adding the glasses direction to the mean of all input images, which starkly showed a distinct glasses addition and subtraction in the images. Then, we experimented with adding and subtracting the glasses direction from reconstructions of real dataset inputs, which added the average “glasses” (which meant sunglasses in our case) to these images. Importantly, subtracting the glasses direction in high magnitudes resulted in a degradation of the eye area, as many glasses images have no eye area (given that it is obscured by the glasses themselves). See figures 14 and 15 for examples. 

##### Figure 14 — Glasses Latent Vector Injection
![Glasses Latent Vector Injection](blog_figures/Glasses_Latent_Vector_Injection.png)
Generated in [Model_1.ipynb](https://colab.research.google.com/drive/1dYU1mzRbGL1LCxkS1nlpzf61fKku0spK#scrollTo=q5rzhnVUdlPu&uniqifier=1).

##### Figure 15 — Glasses Latent Vector Injection Real Faces
![Glasses Latent Vector Injection Real Faces](blog_figures/Glasses_Latent_Vector_Injection_Real_Faces.png)
Generated in [Model_1.ipynb](https://colab.research.google.com/drive/1dYU1mzRbGL1LCxkS1nlpzf61fKku0spK#scrollTo=ZbLs5E-HkOtY&uniqifier=1).

### Unlearning The Latent Vector
Since we could now add and subtract glasses from images, we wanted to see if we could tune the pVAE to unlearn glasses entirely. We trained the pVAE to unlearn the glasses latent vector direction by training the weights towards reconstructing versions of the images that were artificially moved away from the glasses direction (specifically -1.5 * glasses latent vector). This had the effect of unlearning the latent direction of glasses in the encoded space, meaning future generations would not have glasses. See Figure 16. 

##### Figure 16 — Glasses VAE Unlearning
![Glasses VAE Unlearning](blog_figures/Glasses_VAE_Unlearning.png)
Generated in [Model_1.ipynb](https://colab.research.google.com/drive/1dYU1mzRbGL1LCxkS1nlpzf61fKku0spK#scrollTo=WmwI8TfklqeU&uniqifier=1).

### Testing On Individual Celebrities (For Privacy)
Now that we’d proven that it was possible to:
1. Unlearn glasses with classical techniques
2. Extract and modify latent vector directionality of specific features to adjust their presence in decoded images
3. Unlearn concepts using latent vector unlearning

### Classical Methods on New Dataset
Because our current celebA dataset didn’t have labels for individual celebrities (making it impossible to unlearn individual people, which is fundamental to our project), we began with finding a new dataset that included individual celebrity labels (~1700 images, 100 per celebrity). We then separated this process into two parts: AutoEncoder Unlearning and Variational AutoEncoder Unlearning.

### Variational AutoEncoder Unlearning On Celebrity Faces
We applied Gradient Ascent, Random Noise Targeting, and Student/Teacher unlearning methods on the pVAE from earlier fine tuned on this new dataset. We tested the unlearned model’s reconstructions against the original (just trained) model using examples from both images of the unlearned celebrity and images of random other celebrities (to prove non-loss on non-target images). See Figures 6 and 7. 

* Forget loss: 1187.1%
* Retain loss: 643.1%

##### Figure 6 — VAE Gradient Ascent Forget Set (Angelina Jolie)
![VAE GA Forget Set](blog_figures/VAE_GA_Forget-Set.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=Mu5FqWWYPmLI).

##### Figure 7 — VAE Gradient Ascent Retain Set (Other Celebrities)
![VAE GA Retain Set](blog_figures/VAE_GA_Retain-Set.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=Mu5FqWWYPmLI).

Next, Figures 8 and 9 show a visualization of Random Noise Targeting on Angelina Jolie’s samples in the dataset. Compared to Gradient Ascent, the loss on the images we intend to retain is much lower. 

##### Figure 8 — VAE Noise Targeting Forget Set (Angelina Jolie)
![VAE RNT Forget](blog_figures/VAE_RNT_Forget.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=SRHWPTLi60yf).

##### Figure 9 — VAE Noise Targeting Retain Set
![VAE RNT Retain](blog_figures/VAE_RNT_Retain.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=SRHWPTLi60yf).

* Forget loss: 585.1%
* Retain loss: -12.3%

Finally, Figures 10 and 11 are the results of the Student / Teacher method. This was a modification of the “Good Teacher / Bad Teacher” method because instead of a “good teacher” (the initially trained model’s reconstructions without unlearning), we simply compared unlearned reconstructions against the actual encoded images in the dataset. The results of this method show the most promising retain loss when compared to the forget loss. The images of Angelina Jolie are grayed out while the reconstructions of the other data is maintained the most.

##### Figure 10 — VAE Bad Teacher Forget Set (Angelina Jolie)
![VAE Teacher Forget](blog_figures/VAE_Teacher_Forget.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=ZD-mvMT1O89L).

##### Figure 11 — VAE Bad Teacher Retain Set (Other Celebrities)
![VAE Teacher Retain](blog_figures/VAE_Teacher_Retain.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=ZD-mvMT1O89L).

* Forget loss: 879.6%
* Retain loss: -30.7%

While in the beginning of our project we tested an AutoEncoder’s unlearning with our three unlearning techniques, we wanted to finally test the results of a VAE against that of an AE directly as they differ in unlearning specific faces.

##### Figure 13 — VAE Unlearning Method Comparison Chart
![VAE Unlearning Comparison Chart](blog_figures/VAE_Unlearning_Comparison_Chart.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=3nJs4gl0Cjf7).

### Classical AutoEncoder Unlearning On Celebrity Faces
Work for this section is done in celeb_unlearning.ipynb. 
#### Autoencoder architecture
* ResBlock: Applies two convolutions and adds the result back to the input, preventing the gradient from vanishing in deep networks and helps preserve fine details like facial features.
* Encoder: Takes a 64x64x3 image and compresses it: 64 to 32 to 16 to 8 to 4 pixels, while expanding channels 3 to 64 to 128 to 256 to 512. A final fully-connected layer squashes this to a single vector of size LATENT_DIM=256. This vector is the model’s compressed understanding of the face. 
* Decoder: Mirrors encoder in reverse. Takes the latent vector, expands it back through transposed convolutions 4 to 8 to 16 to 32 to 64, ending with a sigmoid that constraints output to [0, 1]. Final output is a reconstructed 64x64 face. 
Autoencoder chains encoder → decoder. The goal is to as accurately as possible reproduce a face. 

#### Loss & Training
Perceptual loss: compares feature representations inside a pretrained VGG16 network, unlike standard MSE loss, which compares pixels directly (produces blurry reconstructions because averaging pixel values is what minimizes MSE). This passes both predicted and target images through three “slices” of VGG: 
* First 4 layers capture layers
* 4-9 capture textures
* 9-16 capture higher-level structure like face shape
The loss is the MSE between these feature maps rather than the pixels themselves, forcing the autoencoder to produce images that “look right”. 

Combined loss and training:
combined_loss adds three terms: MSE (pixel accuracy), perceptual (sharpness), and SSIM (structural similarity – penalizes differences in contrast and luminance patterns). Weights w_mse=1.0, w_perc=0.1, w_ssim=0.5 balance their contributions. 

Training uses AdamW (Adam with weight decay to prevent overfitting) and cosine annealing (learning rate starts at 5e-5 and smoothly decays to near zero by epoch 30, letting the model make big updates early and fine-tune carefully at the end). Gradient clipping at 1.0 prevents any single bad batch from causing a large destabilizing weight update. The model trains on the full dataset (all celebrities) for 100 epochs and saves to models/AE_orig.pt. 
Evaluation
compute_mse evaluates reconstruction quality — lower MSE means more faithful reconstruction. It runs in torch.no_grad() because we don't need gradients for evaluation, saving memory and time.
show_reconstructions visualizes original images in row 1 and their reconstructions in row 2, letting you visually assess quality.

#### Evaluation
compute_mse evaluates reconstruction quality — lower MSE means more faithful reconstruction. It runs in torch.no_grad() because we don't need gradients for evaluation, saving memory and time.
show_reconstructions visualizes original images in row 1 and their reconstructions in row 2, letting you visually assess quality. Figures 17 and 18 show visualizations of our model performance. 

##### Figure 17 — Celebrity AE Model Performance on Forget Set
![Celebrity Face Model Performance on Forget Set](blog_figures/AE_Before_Forget.png)
Generated in [celeb_unlearning.ipynb, Cell 16](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C16).

##### Figure 18 — Celebrity AE Model Performance on Retain Set
![Celebrity Face Model Performance on Retain Set](blog_figures/AE_Before_Retain.png)
Generated in [celeb_unlearning.ipynb, Cell 16](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C16).

* Forget set MSE: 0.01787
* Retain set MSE: 0.02863

#### Key Metrics for Unlearning
Forget set MSE
Retain set MSE
% change in MSE (for each): we want a high % change in MSE on the forget set, but low on the retain set
Forget / Retain MSE ratio: higher = more successful unlearning

##### Gradient Ascent Unlearning
1. Starts with exact copy of original
2. Gradient ascent on the forget set: compute MSE loss between model’s reconstruction and original image, then backpropagate the negative of this loss. (-forget_loss).backward() maximizes it, pushing the model to reconstruct the forget celebrity worse. 
3. Gradient descent on retain set: for 3 steps, compute MSE between unlearn_model’s output and original_model’s output on retain images, and minimize normally. 
4. This runs for 300 steps. W_RETAIN=5.0 means retain preservation is 5x stronger than forgetting. 

Figures 19 and 20 show results. 

##### Figure 19 — Celebrity Gradient Ascent on Forget Set
![Celebrity Gradient Ascent on Forget Set](blog_figures/AE_GA_Forget.png)
Generated in [celeb_unlearning.ipynb, Cell 23](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C23).

##### Figure 20 — Celebrity Gradient Ascent on Retain Set
![Celebrity Gradient Ascent on Retain Set](blog_figures/AE_GA_Retain.png)
Generated in [celeb_unlearning.ipynb, Cell 24](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C24).

| Metric | Before | After | Change |
|---|---|---|---|
| Forget MSE (Angelina Jolie) | 0.01787 | 0.17908 | +902.1% |
| Retain MSE (other celebrities) | 0.02863 | 0.03005 | +5.0% |
Forget/Retain MSE ratio: 5.96

##### Random Noise Unlearning
Structurally identical to gradient ascent, but in the forget step: instead of maximizing reconstruction error with no specific target, the model is trained to output random noise when it sees the forget celebrity’s face. torch.rand_like(f_imgs) generates a fresh random tensor every step, so the model can’t learn any consistent wrong mapping, instead just produce random noise for that identity. Figures 21 and 22 show results. 

##### Figure 21 — Celebrity Random Noise Targeting on Forget Set
![Celebrity Random Noise Targeting on Forget Set](blog_figures/AE_RN_Forget.png)
Generated in [celeb_unlearning.ipynb, Cell 30](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C30).

##### Figure 22 — Celebrity Random Noise Targeting on Retain Set
![Celebrity Random Noise Targeting on Retain Set](blog_figures/AE_RN_Retain.png)
Generated in [celeb_unlearning.ipynb, Cell 29](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C29).

| Metric | Before | After | Change |
|---|---|---|---|
| Forget MSE (Angelina Jolie) | 0.01787 | 0.06514 | +264.5% |
| Retain MSE (other celebrities) | 0.02863 | 0.03288 | +14.8% |
Forget/Retain MSE ratio: 1.98

##### Good Teacher / Bad Teacher Unlearning
Created and intentionally undertrained BadTeacher, which is a shallow 4-layer autoencoder trained for 3 epochs on the forget set. Produces blurry, low-quality reconstructions on purpose, then frozen (See Figure 23).
* For forget set images: the student (bt_model) is trained to match bad_teacher’s output
* For retain set images: the student is trained to match good_teacher (original_model)
Learns specific, deterministic degraded mapping. 

Figures 24 and 25 show our results for this method. 

##### Figure 23 — Celebrity Bad Teacher
![Celebrity Bad Teacher](blog_figures/AE_Teacher.png)
Generated in [celeb_unlearning.ipynb, Cell 33](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C33).

##### Figure 24 — Celebrity Good / Bad Teacher Targeting on Forget Set
![Celebrity Good / Bad Teacher on Forget Set](blog_figures/AE_Teacher_Forget.png)
Generated in [celeb_unlearning.ipynb, Cell 38](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C38).

##### Figure 25 — Celebrity Good / Bad Teacher on Retain Set
![Celebrity Good / Bad Teacher on Retain Set](blog_figures/AE_Teacher_Retain.png)
Generated in [celeb_unlearning.ipynb, Cell 37](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C37).

| Metric | Before | After | Change |
|---|---|---|---|
| Forget MSE (Angelina Jolie) | 0.01787 | 0.07268 | +306.7% |
| Retain MSE (other celebrities) | 0.02863 | 0.03503 | +22.3% |
Forget/Retain MSE ratio: 2.08

### Analysis
Of these three methods, the gradient ascent method was most effective in both forgetting Angelina Jolie’s face and with retaining the quality of other celebrity faces. Overall, good teacher bad teacher did the next best, and random noise targeting performed the worst. However, good teacher bad teacher was better at forgetting the forget set, while random noise targeting was better at retaining the retain set. Figures 26 and 27 show visualizations of all methods together. 

##### Figure 26 — Celebrity Unlearning Comparisons on Forget Set
![Celebrity Unlearning Comparisons on Forget Set](blog_figures/AE_Unlearning_Comparison_F.png)
Generated in [celeb_unlearning.ipynb, Cell 40](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C40).

##### Figure 27 — Celebrity Unlearning Comparisons on Retain Set
![Celebrity Unlearning Comparisons on Forget Set](blog_figures/AE_Unlearning_Comparison_R.png)
Generated in [celeb_unlearning.ipynb, Cell 41](https://vscode.dev/github/RexTabachnick/Machine-Unlearning-In-Applied-Contexts/blob/main/celeb_unlearning.ipynb#C41).

#### Experimentation with Latent Vector Spaces
At this moment, since we knew that we could artificially inject the glasses latent direction into encoded images, we wanted to experiment with obtaining the latent vector of a single person. For privacy compliance, this would allow us to remove someone’s facial direction from the latent space, thereby removing their face entirely from being generated. However, due to the constraints of our dataset, we’d had only 100 images from each celebrity, which was not enough to construct a direction that looked like our subject (Angelina Jolie). While we did unlearn her face using the classical unlearning techniques, latent vector injection was not clear enough to show consistent results. We did experiment with overfitting on Angelina’s face to collapse the latent space towards narrower results on the VAE, however the images were so different from each other that the face still did not emerge. (See Figure 12). 

##### Figure 12 — VAE Latent Vector Injection (Celebrity Face)
![VAE Latent Vector Injection Celebrity Face](blog_figures/VAE_Latent-Vector-Injection_Celebrity-Face.png)
Generated in [VAE_On_Individual_Celebrities.ipynb](https://colab.research.google.com/drive/1XHcxZ6gEbUwu-8ir8mK600_5V3pONTJZ#scrollTo=-kCZXWhPvkpy).

## Discussion
As a result of our experiments, we learned a lot about the process of unlearning and its complexities. Unlearning is not a deterministic process because it is not possible (currently) to reverse the process of machine learning. However, we can apply various methods to get close to the ideal result. Gradient Ascent, Random Noise Targeting, and Good Teacher / Bad Teacher techniques for unlearning provided promising results that degraded concepts like glasses and individual people.

### Problems we ran into:
1. Posterior collapse on the from-scratch VAE: 
Our first approach was to train a VAE from scratch on the celebrity dataset (~1,800 images), which produced blurry average faces with no individual identity. This is posterior collapse: a VAE balances two competing goals (accurate reconstruction and an organized latent space), and with two little data it takes the lazy route. It stops using the latent code and outputs an average face for everything. We switched to a pretrained VAE. 

2. Eye-region degradation in glasses unlearning: 
The decoder fine-tuning method reduced glasses in reconstructions but damaged the eye region. Eyes appeared smudged, uneven, or unnaturally shaped. Even though we used CelebA’s clean glasses labels, the direction we extracted is the difference of the two mean latents, which captures every visual pattern that systematically differs between glasses-wearers and non-wearers in the data - not just glasses. Glasses wearers in the training set tend to be photographed with their eyes partially obscured by frames or lenses, so the “glasses direction” carries some of that obstructed-eye signal too. Subtracting it takes out both, and the decoder fills in the eye region poorly. 

3. Failed identity direction extraction: 
Before applying the three classical methods to identity, we tried the latent arithmetic approach from stage 2: computing a specific celebrity direction as the mean latent of their images minus the mean latent of the rest of the celebrity images. The result was unusable: adding it to the other faces did not produce recognizable features for the specific celebrity. The cause was sample size. The glasses direction was estimated from 13,000 images while the celebrity with the most images only had about 100. With that few samples, the mean is dominated by incidental photo-shoot features (lighting, pose, hair, background) rather than identity. This confirmed our entanglement hypothesis: identity follows a more regional shape in a latent space rather than a clean directional axis. 

## Overarching Research Field
Our project sits at the small-scale end of a much larger research field. The closest production-scale comparison is Erased Stable Diffusion, which fine-tunes Stable Diffusion to remove named celebrities and artistic styles. Its self-distillation setup is conceptually similar to our good-teacher/bad-teacher method. The latent vector arithmetic technique we test is similar to DCGAN and InterFaceGAN, which showed that binary attributes correspond to linear directions in face-model latent spaces. The other three methods come from the classifier unlearning literature (Cao & Yang, 2015). Our project consists of adapting all four to generative reconstruction and comparing them on both an attribute and an identity target. 

## Future Work
In the future, we would like to experiment more with latent vector injection. We believe that with sufficient data, we could add and subtract faces from each other and from themselves, which has a number of real-world interpretations. Similarly, our biggest find was the surprising degree to which injecting the ‘glasses’ latent vector direction resulted in clear glasses emerging into images. Injecting the glasses latent direction onto our reconstructions caused glasses to emerge on their faces while removing them tended to degrade the eye area considerably. This is because glasses often cover the eye area, which degrades the data in that region of the face. 

In the future, with more time, testing on images of individual people that are not part of the original dataset is important because it proves that their faces are being unlearned instead of the input images themselves. Gaining access to more data on individual people would be paramount to this task.

We would also like to experiment further with classical unlearning techniques, as privacy compliance is an increasingly sensitive issue as the large language models require increasing amounts of data. If data on peoples’ lives becomes more valuable, the ability to freely remove certain data is paramount to their security and their freedom. 

## Conclusion
Our project demonstrated further the need for research on unlearning. Many companies today use similar algorithms to the ones that we used. Financial companies will use CNNs and Transformers to calculate risk and adapt their pricing person-to-person. If they use non-compliant data, they are incentivized to do nothing, as otherwise they would have to spend lots of time and money rebuilding their model without the sensitive data. With verifiable unlearning techniques, these companies could subtract data as-is, incentivising them to be maximally compliant with privacy regulations. This would allow people to freely determine whether they would want their data out in public and it would make it much easier for companies to comply with it.

## Code
The code for this project is availble in this repository (https://github.com/RexTabachnick/Machine-Unlearning-In-Applied-Contexts). To see commit history of celeb_unlearning.ipynb, visit https://github.com/jasminesun1/machine_unlearning. 