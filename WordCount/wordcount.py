# Python 代码示例：使用提供的文本生成词云

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 使用前面的文本
text = """
        Scala combines object-oriented and functional programming in one concise, high-level language. 
        Scala's static types help avoid bugs in complex applications, and its JVM and JavaScript runtimes 
        let you build high-performance systems with easy access to huge ecosystems of libraries. Scala classes 
        are ultimate containers of data and behavior, similar to Java's classes. Traits in Scala are abstract data 
        types containing certain fields and methods. Multiple traits can be combined. Scala's case classes are used 
        for pattern matching, which is a very powerful feature. Implicit parameters and conversions enrich existing 
        libraries. Scala also has a vibrant community and rich ecosystem which includes frameworks like Play for web 
        applications, Akka for concurrent processing, and Apache Spark for big data.
        """

# 创建词云对象
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10).generate(text)

# 展示词云
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

