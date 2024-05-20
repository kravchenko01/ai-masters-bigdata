CREATE TABLE IF NOT EXISTS Kravchenko01.ratings(
    user_id INT,
    movie_is INT,
    rating INT,
    tmpstmp INT)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    ESCAPED BY '\"'
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE;

SELECT * FROM Kravchenko01.ratings;
