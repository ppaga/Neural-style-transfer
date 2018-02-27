# Neural Style
This is an implementation of the Neural Style paper in tensorflow (using tensorlayer and keras to build the vgg network). 
I tried to follow the Gatys et al. paper, but some details may differ.

As it turns out, adding a total variation loss helps reduce the bright color patchwork that tended to be present. The strength of that regularization can be modulated via the TV_scaling parameter, I've found 1e-3 to be pretty good.
