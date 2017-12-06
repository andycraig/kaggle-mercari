library(tidyverse)
library(stringr)

# Load train and create variables for different category levels.

train = read_tsv("../data/train.tsv") %>%
  mutate(category_name_1 = str_match(category_name, "(.*?)/.*")[,2],
         category_name_2 = str_match(category_name, "(.*)/.*")[,2]) %>%  
  rename(category_name_3 = category_name)
  
# How many rows?

nrow(train)

# What columns?

names(train)

# Head.

head(train)

# Histogram of item_condition_id

plot(hist(train$item_condition_id))

# Mostly 1, 2, 3. Few 4, 5.

# How many category names?

train$category_name_1 %>% unique() %>% length()

# 11

train$category_name_2 %>% unique() %>% length()

# 145

train$category_name_3 %>% unique() %>% length()

# 1288

# Test dataset.

test = read_tsv("../data/test.tsv")%>% 
  mutate(category_name_1 = str_match(category_name, "(.*?)/.*")[,2],
         category_name_2 = str_match(category_name, "(.*)/.*")[,2]) %>% 
  rename(category_name_3 = category_name)

# Any category_name in test but not train?

test$category_name_1 %>% unique() %>% setdiff(train$category_name_1 %>% unique())

# 0

test$category_name_2 %>% unique() %>% setdiff(train$category_name_2 %>% unique())

# 0

test$category_name_3 %>% unique() %>% setdiff(train$category_name_3 %>% unique())

# 23

# So can use category_name_2 for test set when category_name doesn't exist in train.

# price by category_name.
train %>% ggplot(aes(x = category_name_2, y = log(price))) +
  geom_point()

# Lots of price variation within given category_name.

# How many items with brand names?

sum(!is.na(train$brand_name)) / length(train$brand_name)

# 57%

# How many unique brand names?

train$brand_name %>% unique() %>% length()

# 4810

# Shipping?

train$shipping %>% unique()

# Binary.

# item_description?

train$item_description[1:5]

# Length?

item_description_lengths = purrr::map(train$item_description, stringr::str_length) %>% unlist() 

item_description_lengths %>% max(na.rm = TRUE)

# 1046

# What is this?

train[which(item_description_lengths == 1046),]

# List of items with their individual prices.

plot(hist(item_description_lengths))

# Try predicting price by category_name and item_condition_id.
# Not enough memory to use category_name_3 so use category_name_2.
# For now, drop category_name NA.

model_lm = train %>% 
  filter(!is.na(category_name_3)) %>% 
  lm(log(price) ~ category_name_2 + item_condition_id, data = .)
