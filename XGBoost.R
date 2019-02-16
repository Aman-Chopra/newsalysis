# library(dplyr)
# library(readr)
# 
# df <- list.files(path="/home/ashraydimri/Personal/News_Paper/Data/",full.names = TRUE) %>% 
#   lapply(read_csv) %>% 
#   bind_rows 
# filenames <- list.files(path = "/home/ashraydimri/Personal/News_Paper/Data/",full.names=TRUE)


library(tm)
df<-read.csv("/home/ashraydimri/Personal/News_Paper/News_Final.csv",stringsAsFactors = F)
#ap_lda <- LDA(df, k = 2, control = list(seed = 1234))



cp<-Corpus(VectorSource(df$Headline))
inspect(cp)


docs <-tm_map(cp,content_transformer(tolower))


toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " ", x))})
docs <- tm_map(docs, toSpace, '-')
docs <- tm_map(docs, toSpace, '’')
docs <- tm_map(docs, toSpace, '‘')
docs <- tm_map(docs, toSpace, '•')
docs <- tm_map(docs, toSpace, '”')
docs <- tm_map(docs, toSpace, '“')
# docs <- tm_map(docs, toSpace, '.')
# docs <- tm_map(docs, toSpace, "'")
# docs <- tm_map(docs, toSpace, '"')
# docs <- tm_map(docs, toSpace, '-')

docs <- tm_map(docs, removePunctuation)
#Strip digits
docs <- tm_map(docs, removeNumbers)
#remove stopwords
docs <- tm_map(docs, removeWords, stopwords('english'))
#remove whitespace
docs <- tm_map(docs, stripWhitespace)
#Good practice to check every now and then
writeLines(as.character(docs[[30]]))
#Stem document
docs <- tm_map(docs,stemDocument)

# docs <- tm_map(docs, content_transformer(gsub),
#                pattern = "organiz", replacement = "organ")
# docs <- tm_map(docs, content_transformer(gsub),
#                pattern = "organis", replacement = "organ")
# docs <- tm_map(docs, content_transformer(gsub),
#                pattern = "andgovern", replacement = "govern")
# docs <- tm_map(docs, content_transformer(gsub),
#                pattern = "inenterpris", replacement = "enterpris")
# docs <- tm_map(docs, content_transformer(gsub),
#                pattern = "team-", replacement = "team")
#define and eliminate all custom stopwords
#myStopwords <- c("can", "say","one","way","use",
                  # "also","howev","tell","will",
                  # "much","need","take","tend","even",
                  # "like","particular","rather","said",
                  # "get","well","make","ask","come","end",
                  # "first","two","help","often","may",
                  # "might","see","someth","thing","point",
                  # "post","look","right","now","think","‘ve ",
                  # "‘re ","anoth","put","set","new","good",
                  # "want","sure","kind","larg","yes,","day","etc",
                  # "quit","sinc","attempt","lack","seen","awar",
                  # "littl","ever","moreov","though","found","abl",
                  # "enough","far","earli","away","achiev","draw",
                  # "last","never","brief","bit","entir","brief",
                  # "great","lot")
#docs <- tm_map(docs, removeWords, myStopwords)


dtm <- DocumentTermMatrix(docs)
#convert rownames to filenames
rownames(dtm) <- df$IDLink
#collapse matrix by summing over columns
freq <- colSums(as.matrix(dtm))
#length should be total number of terms
length(freq)
#create sort order (descending)
ord <- order(freq,decreasing=TRUE)
#List all terms in decreasing order of freq and write to disk
#freq[ord]
write.csv(freq[ord],"word_freq.csv")


library(topicmodels)

burnin <- 10000
iter <- 4000
thin <- 800
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

k <- 10

rowTotals <- apply(dtm , 1, sum)
dtm.new   <- dtm[rowTotals> 0, ] 


ldaOut <-LDA(dtm,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))

ldaOut.terms <- as.matrix(terms(ldaOut,6))
write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))

topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))

topic1ToTopic2 <- lapply(1:nrow(dtm),function(x)sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])
topic2ToTopic3 <- lapply(1:nrow(dtm),function(x)
  sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])
write.csv(topic1ToTopic2,file=paste("LDAGibbs",k,"Topic1ToTopic2.csv"))
write.csv(topic2ToTopic3,file=paste("LDAGibbs",k,"Topic2ToTopic3.csv"))


#################################
df2<-df[df$Headline!="",]
ldares<-as.data.frame(ldaOut.topics)

ldares$idlink<-rownames(ldares)
colnames(ldares)[1]<-"Topics_lda"
finalres<-merge(x=df2,y=ldares,by.x = "IDLink", by.y = "idlink",all.x = T)
write.csv(finalres,"/home/ashraydimri/Personal/News_Paper/finalResultAfterTopicsFromLDA.csv")



#################################
df2<-df[df$Title!="",]
ldares<-as.data.frame(ldaOut.topics)

ldares$idlink<-rownames(ldares)
colnames(ldares)[1]<-"Topics_lda_title"
finalres<-merge(x=df2,y=ldares,by.x = "IDLink", by.y = "idlink",all.x = T)

dfold<-read.csv("/home/ashraydimri/Personal/News_Paper/finalResultAfterTopicsFromLDA.csv",stringsAsFactors = F)
finalres2<-finalres[,c(1,12)]
resLDA<-merge(x=dfold,y=finalres2,by.x = "IDLink", by.y = "IDLink",all.x = T)
resLDA$X<-NULL
write.csv(resLDA,"/home/ashraydimri/Personal/News_Paper/finalResultAfterTopicsFromLDAv1.1.csv",row.names = F)