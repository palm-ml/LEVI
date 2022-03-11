function dist = testModel(prediction,label,target)
    dist = [];
    dist(1) = RankingLoss(prediction',target');
    dist(2) = OneError(prediction',target');
    dist(3) = Coverage(prediction',target');
    dist(4) = HammingLoss(label',target');
    dist(5) = AveragePrecision(prediction',target');