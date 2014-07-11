#!/usr/bin/env Rscript
library(ggplot2)
library(gridExtra)
data <- read.csv("~/Projects/2014-07-09_3body/data.csv")
b = ggplot(data)
ptsize <- 0.8
print(
  grid.arrange(
    b + geom_point(aes(r, vr), size=ptsize)
    , b + geom_point(aes(z, vz), size=ptsize)
    , b + geom_point(aes(y, vy), size=ptsize)
    , b + geom_point(aes(w, theta), size=ptsize)
    , b + geom_point(aes(t, r), size=ptsize)
    , b + geom_point(aes(t, z), size=ptsize)
    , b + geom_point(aes(t, y), size=ptsize)
    , b + geom_point(aes(t, theta), size=ptsize)
    , ncol=2)
  )

