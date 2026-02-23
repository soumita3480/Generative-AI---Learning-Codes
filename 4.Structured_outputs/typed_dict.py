from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

# person is the dict name and Person says the data should be of typeddict
person: Person= {
    # 'name':'Soumita','Age':'35' # this will also work since typeddict doesnot validate the datatype of data passed as i/p
    'name':'Soumita','Age':35
}

print(person)
