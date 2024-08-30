# Projeto de NTLK
import nltk
from nltk.corpus import mac_morpho
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tokenize import word_tokenize

# Baixar os recursos necessários do NLTK, caso ainda não tenha baixado
nltk.download('mac_morpho')
nltk.download('punkt')
nltk.download('punkt_tab')

# Carregar o corpus MacMorpho
corpus = mac_morpho.tagged_sents()

# Dividir o corpus em treino (80%) e teste (20%)
train_size = int(len(corpus) * 0.8)
train_sents = corpus[:train_size]
test_sents = corpus[train_size:]

# Treinar um modelo de etiquetagem de palavras utilizando Unigram, Bigram e Trigram
unigram_tagger = UnigramTagger(train_sents)
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
trigram_tagger = TrigramTagger(train_sents, backoff=bigram_tagger)

# Avaliar a precisão do modelo no conjunto de teste
accuracy = trigram_tagger.evaluate(test_sents)
print(f'Precisão do modelo: {accuracy:.4f}')

# Exemplo de anotação de um texto em português brasileiro
texto = "Eu gosto de aprender sobre processamento de linguagem natural."

# Tokenizar o texto
tokens = word_tokenize(texto, language='portuguese')

# Realizar a etiquetagem (POS tagging) utilizando o modelo treinado
tags = trigram_tagger.tag(tokens)

# Exibir os tokens com suas respectivas etiquetas
print("Anotação do texto:")
for token, tag in tags:
    print(f'{token} -> {tag}')

# Função para obter estatísticas sobre as etiquetas do corpus
def estatisticas_etiquetas(tagged_sents):
    tags = [tag for sent in tagged_sents for _, tag in sent]
    freq_dist = nltk.FreqDist(tags)
    print("Distribuição de frequência das etiquetas:")
    for tag, freq in freq_dist.most_common():
        print(f'{tag}: {freq}')

# Exibir estatísticas sobre as etiquetas do conjunto de treino
estatisticas_etiquetas(train_sents)