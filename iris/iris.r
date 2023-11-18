if(!require(data.table)) { install.packages("data.table"); library(data.table) }
if(!require(magrittr)) { install.packages("magrittr"); library(magrittr) }
if(!require(tidyr)) { install.packages("tidyr"); library(tidyr) }
if(!require(ggplot2)) { install.packages("ggplot2"); library(ggplot2) }
if(!require("scales")) { install.packages("scales"); library("scales") }
theme_set(theme_bw())

# nn
umbrann = as.data.table(read.csv(file="gd_nn.csv",header=TRUE,sep = ","))[name!="Operator"]
numpynn = as.data.table(read.csv(file="numpy_nn.csv",header=TRUE,sep = ","))
duckdbnn = as.data.table(read.csv(file="duckdb_iris.csv",header=TRUE,sep = ","))
psqlnn  = as.data.table(read.csv(file="psql_nn.csv",header=TRUE,sep = ","))

data2nn  = rbind(umbrann[threads==8,c("name", "atts", "limit", "lr", "iter", "execution_time")], numpynn[,c("name", "atts", "limit", "lr", "iter", "execution_time")], duckdbnn[,c("name", "atts", "limit", "lr", "iter", "execution_time")], psqlnn[,c("name", "atts", "limit", "lr", "iter", "execution_time")])
iris = ggplot(data2nn, aes(x=limit, y=execution_time, color=name, linetype=name)) + facet_grid(iter~atts, scales = "free") +
  geom_line() + xlab("Tuples") + ylab("Execution Time [s]") + theme(legend.position = "bottom")+ labs(color='', linetype='') + scale_x_continuous(trans='log10') + scale_y_continuous(trans='log10', labels=comma)

ggsave("iris.pdf", device = cairo_pdf, plot = iris, units = "in", width = 8, height = 3)
