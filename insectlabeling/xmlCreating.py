from os import listdir, makedirs
from os.path import isfile, join, split, exists

import pandas as pd
from cornerDetect import detect_with_corner
from scipy import stats

from insectlabeling.pascal_voc_io import PascalVocWriter


class InsectDetector(object):


    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.dirname = split(path)[-1]
        self.filelist = [f for f in listdir(path) if isfile(join(path, f))]

    def wholeFolderConcat(self, test=None, neighbors=30, insectN=6):
        """
        It finds insects iterating whole files in given directory path and
        returns concated Dataframe for whole objects.

        :param test: If you change the codes, and you want to just try few number of samples, then put the numbers of
        how much images you will put in this iteration.
        :param neighborsN: To detect corner, it apply KNN algorithm.
        We should find proper neighbors numbers adjusting this parameter. If considering object is big, then it needs
        to increase neighborsN because if we decrease N, then it will provides you several figures on one object.s
        :param insectsN: How much insects on one photo, We will sort and extract contours considering this number.

        :return: A dataframe covering whole photos of given directory.
        """
        flag = 0
        total = len(self.filelist)
        df = []

        for f in self.filelist:
            filepath = join(self.path, f)
            df.append(detect_with_corner(filepath, neighbors, insectN))
            flag += 1
            print("Current progress: %d / %d" % (flag, total))
            if test:
                if test == flag: break

        df = pd.concat(df, ignore_index=True)

        return df

    def imageLabel(self, df):
        """

        :param df:
        :return:
        """

        for f in df.path.unique():
            first = df[df.path == f][:1]
            shape = first.iloc[0]['shape']
            rects = df[df.path == f]['rects']
            image = split(f)[-1]

            writer = PascalVocWriter(self.dirname, image, shape, localImgPath=f)

            for (x, y, w, h) in rects:
                writer.addBndBox(x, y, x+w, y+h, self.name, 0)


            targetpath = self.path + '/results/'

            if not exists(targetpath):
                makedirs(targetpath)

            targetfile = targetpath + image.split('.')[0] + '.xml'
            writer.save(targetFile=targetfile)

        print('Finished!')

    def zscoreCal(self, df, z=1.5, max=None, min=None):

        df = df[(df.area < max) & (df.area > min)]
        df['zscore'] = abs(stats.zscore(df['area']))
        df = df[df.zscore < z]

        return df



if __name__ == '__main__':
    detector = InsectDetector('/home/paul/Downloads/insects/8_diabrotica_virgifera', 'diabrotica_virgifera')

    df = detector.wholeFolderConcat()
    df.to_excel('./result2.xlsx')
    df = detector.zscoreCal(df, z=1.5, max=20000, min=2000)
    detector.imageLabel(df)