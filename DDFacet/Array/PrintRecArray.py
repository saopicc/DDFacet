def remove_field_name(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b

from prettytable import PrettyTable


def Print(CatIn, RemoveFieldName='ChanFreq'):
    if RemoveFieldName in CatIn.dtype.names:
        Cat = remove_field_name(CatIn, RemoveFieldName)
    else:
        Cat = CatIn

    x = PrettyTable(Cat.dtype.names)
    for row in Cat:
        x.add_row(row)
    print x
