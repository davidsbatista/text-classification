from collections import defaultdict

import codecs
import pandas as pd


def get_genres():
    genres = defaultdict(list)
    unique_genres = set()
    with codecs.open("genres_utf8.list", "r") as f:
        for line in f:
            if line.startswith('"'):
                data = line.split('\t')
                movie = data[0]
                genre = data[-1].strip()
                genres[movie].append(genre)
                unique_genres.add(genre)
    return genres, sorted(unique_genres)


def get_plots():
    with codecs.open("plot_utf8.list", "r") as f:
        data = []
        inside = False
        plot = ''
        full_title = ''
        for line in f:
            if line.startswith("MV:") and not inside:
                inside = True
                full_title = line.split("MV:")[1].strip()

            elif line.startswith("PL:") and inside:
                plot += line.split("PL:")[1].replace("\n", "")

            elif line.startswith("MV:") and inside:
                short_title = full_title.split('{')[0].strip()
                data.append((short_title, full_title, plot))
                plot = ''
                inside = False
    return data


def main():

    # for TV shows plots contain names and episodes, e.g.:
    #   "#LawstinWoods" (2013) {The Case of the Case (#1.5)}
    #
    # genres contain only main title, e.g.
    #   #LawstinWoods"  (2013)        Sci-Fi

    genres, unique_genres = get_genres()
    data = []
    for movie in genres:
        row = [0]*len(unique_genres)
        for g in genres[movie]:
            row[unique_genres.index(g)] = 1
        row.insert(0, movie)
        data.append(row)

    genres_df = pd.DataFrame(data)
    genres_df.columns = ['short_title'] + unique_genres
    print genres_df.shape

    plots = get_plots()
    plots_df = pd.DataFrame(plots)
    plots_df.columns = ['short_title', 'title', 'plot']
    print plots_df.shape

    data_df = plots_df.merge(genres_df, how='inner', on='short_title')
    data_df.dropna(inplace=True)
    data_df.drop('short_title', axis=1, inplace=True)

    # 'Sci-fi' is not associated with any plot
    data_df.drop('Sci-fi', axis=1, inplace=True)
    print data_df.shape

    data_df.to_csv(path_or_buf='movies_genres.csv', sep='\t',
                   header=True, encoding='utf8', index=False)

if __name__ == "__main__":
    main()
