this.dir = dirname(parent.frame(2)$ofile)
setwd(this.dir)

library(dplyr)
library(data.table)

# load with less memory usage
train <- fread('../input/train.csv', 
         select = c('Semana', 'Cliente_ID', 'Producto_ID', 
                    'Demanda_uni_equil'),
         header = T, sep = ',', data.table = F)
train <- rename(train, Demanda = Demanda_uni_equil)

# load if more memory is available
# train <- fread('../input/train.csv', header = T, sep = ',', data.table = F)
# names(train) <- c('week', 'depot_ID', 'channel_ID', 'route_ID', 'client_ID',
#                    'product_ID', 'salesx', 'salesPesosx', 'returnsx',
#                    'returnsPesosx', 'adjDemx')

wk <- 3
for (lagName in c('_w3', '_w4', '_w5', '_w6', '_w7', '_w8', '_w9')) { # time lagged column names
  addDF <- filter(train, Semana == wk) %>% select(-Semana)
  train <- filter(train, Semana != wk) # don't need this week now (save memory) 
  colnames(addDF) <- gsub('x$', lagName, colnames(addDF)) # change colnames to "lagged" names
  if (wk == 3) {
    featDF <- addDF
  } else {
    featDF <- full_join(featDF, addDF) 
  }
  wk <- wk + 1
}

head(featDF)

write.csv(featDF[1:10,], 'train_df.csv', row.names = FALSE)
