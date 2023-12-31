if(!require(data.table)) { install.packages("data.table"); library(data.table) }
if(!require(magrittr)) { install.packages("magrittr"); library(magrittr) }
if(!require(tidyr)) { install.packages("tidyr"); library(tidyr) }
if(!require(ggplot2)) { install.packages("ggplot2"); library(ggplot2) }
if(!require("scales")) { install.packages("scales"); library("scales") }
theme_set(theme_bw())

umbramnist = as.data.table(read.csv(file="gd_mnist.csv",header=TRUE,sep = ","))[name!="Operator"]
numpymnist = as.data.table(read.csv(file="numpy_mnist.csv",header=TRUE,sep = ","))
duckdbmnist = as.data.table(read.csv(file="duckdb_mnist.csv",header=TRUE,sep = ","))
datamnist  = rbind(umbramnist[,c("name", "atts", "limit", "lr", "iter", "execution_time")], numpymnist[,c("name", "atts", "limit", "lr", "iter", "execution_time")], duckdbmnist[,c("name", "atts", "limit", "lr", "iter", "execution_time")])#, psqlnn[,c("name", "atts", "limit", "lr", "iter", "execution_time")])
mnist = ggplot(datamnist[limit>2], aes(x=limit, y=execution_time, color=name, linetype=name)) + facet_wrap(~atts, scales = "free") + 
  geom_line() + xlab("Batch Size") + ylab("Execution Time [s]") + theme(legend.position = "bottom")+ labs(color='', linetype='') + scale_x_continuous(trans='log10') + scale_y_continuous(trans='log10', labels=comma)

ggsave("mnist.pdf", device = cairo_pdf, plot = mnist, units = "in", width = 4, height = 3)

