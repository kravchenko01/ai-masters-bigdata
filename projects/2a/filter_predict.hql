ADD FILE 2a.joblib;
ADD FILE projects/2a/model.py;
ADD FILE projects/2a/predict.py;

INSERT OVERWRITE TABLE hw2_pred 
    SELECT TRANSFORM(*)
    USING 'predict.py'
    AS id, pred
    FROM hw2_test
    WHERE if1 IS NOT NULL AND if1 > 20 AND if1 < 40;

-- SELECT * FROM hw2_test
-- WHERE if1 is not NULL AND if1 > 20 AND if1 < 40;
