library(data.table)
library(magrittr)
library(tidyr)
library(ggplot2)
library(tikzDevice)
library("scales")
theme_set(theme_bw())
library(tikzDevice)

# nn
umbrann = as.data.table(read.csv(file="gd_nn.csv",header=TRUE,sep = ","))[name!="Operator"]
numpynn = as.data.table(read.csv(file="numpy_nn.csv",header=TRUE,sep = ","))
psqlnn  = as.data.table(read.csv(file="psql_nn.csv",header=TRUE,sep = ","))

data2nn  = rbind(umbrann[threads==8,c("name", "atts", "limit", "lr", "iter", "execution_time")], numpynn[,c("name", "atts", "limit", "lr", "iter", "execution_time")], psqlnn[,c("name", "atts", "limit", "lr", "iter", "execution_time")])
iris = ggplot(data2nn, aes(x=limit, y=execution_time, color=name, linetype=name)) + facet_grid(iter~atts, scales = "free") +
  geom_line() + xlab("Tuples") + ylab("Execution Time [s]") + theme(legend.position = "bottom")+ labs(color='', linetype='') + scale_x_continuous(trans='log10') + scale_y_continuous(trans='log10', labels=comma)
iris

tikz(file = "./iris.tex", width = 5, height = 3.5)
show(iris)
dev.off()
