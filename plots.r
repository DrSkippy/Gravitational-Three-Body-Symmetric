#!/usr/bin/env Rscript
library(ggplot2)
library(gridExtra)
data <- read.csv("./data.csv")
b = ggplot(data)
ptsize <- 0.2
png(filename="./phase_space.png", width=700, height=1200)
print(
  grid.arrange(
    b + geom_point(aes(r, vr), size=ptsize)
    , b + geom_point(aes(z, vz), size=ptsize)
    , b + geom_point(aes(y, vy), size=ptsize)
    , b + geom_point(aes(theta, w), size=ptsize)
    , b + geom_point(aes(t, r), size=ptsize)
    , b + geom_point(aes(t, z), size=ptsize)
    , b + geom_point(aes(t, y), size=ptsize)
    , b + geom_point(aes(theta, z), size=ptsize)
    , ncol=2)
  )
dev.off()

data1 <- read.csv("./returnmap.csv")
b1 = ggplot(data1)
ptsize <- 1.0
png(filename="./yearly.png", width=700, height=1200)
print(
  grid.arrange(
    b1 + geom_point(aes(t, period), size=ptsize)
    , b1 + geom_point(aes(t, z), size=ptsize)
    , b1 + geom_point(aes(t, r), size=ptsize)
    , b1 + geom_point(aes(z, vz), size=ptsize)
    , ncol=1)
  )
dev.off()
