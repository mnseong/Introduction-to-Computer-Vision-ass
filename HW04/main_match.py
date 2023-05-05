import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/book', ratio_thres=0.6)
    im = utils.Match('./data/scene', './data/box', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)
    # plt.savefig('scene_box.png')

    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
    plt.title('MatchRANSAC')
    plt.imshow(im)
    # plt.savefig('library_library2.png')

if __name__ == '__main__':
    main()
