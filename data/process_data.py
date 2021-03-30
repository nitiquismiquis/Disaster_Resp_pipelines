import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    messages_filepath = 'data/disaster_messages.csv'
    categories_filepath = 'data/disaster_categories.csv'

    #Reads disaster_messages.csv and drop the original column
    df_mess = pd.read_csv(messages_filepath, encoding='latin-1')
    df_mess.drop(['original'],axis=1,inplace=True)
    
    #Reads disaster_categories.csv
    df_cat = pd.read_csv(categories_filepath, encoding='latin-1')

    # Merges both dataframes on ['Id']
    df = df_mess.merge(df_cat, how='outer', on=['id'])
    
    return df


def clean_data(df):
    ### Creates columns with correspondent values of the 'categories' column

    # Provides a list with all the columns extracted from the category column
    cat = df.loc[0,'categories']
    cat_list = cat.split(';')
    col_names = []
    for val in cat_list:
        c = val.split('-')[0]
        col_names.append(c)

    # Creates all columns in df with correct value
    for col in col_names[0:-1]:
        try:
            df[col]          = df['categories'].apply(lambda st: st[st.find("-")+1:st.find(";")])
            df['categories'] = df['categories'].str.split(';',n=1).str[1:]
            df['categories'] = df['categories'].apply(lambda x: str(x[0]))
        # deals with the last column
        except:
            df[col_names[-1]]= df['categories'].apply(lambda st: st[st.find("-")+1:])
    
    # Drops de 'categories' column
    df.drop(['categories'], axis = 1, inplace = True)
    
    #Remove duplicates
    print('Number of columns: {}, and number of duplicates: {}'.format(df['message'].shape[0],df[df.duplicated() == True]['id'].count()))
    print('The following rows are duplications:')
    print(df[['id','message']][df.duplicated()==True])
    df.drop_duplicates(inplace=True)
    #df.drop_duplicates(subset=['id', 'message'],inplace=True)
    print('Number of columns of new dataframe excluding duplicates: {}'.format(df['message'].shape[0]))
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()