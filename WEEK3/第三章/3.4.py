class Node(object):
    def __init__(self,item):
        self.item = item
        self.next = None
class LinkList(object):
    def __init__(self):
        self.head = None
    def isempty(self):
        return self.head is None
    def items(self):
        cur = self.head
        while cur is not None:
            yield cur.item
            cur = cur.next
    def find(self,item):
        return item in self.items()
    def length(self):
        cur = self.head
        cnt = 0
        while cur is not None:
            cnt += 1
            cur = cur.next
        return cnt
    def append(self,item):
        node = Node(item)
        if self.isempty():
            self.head = node
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            cur.next = node
    def delete(self,item):
        if not self.find(item):
            return "?"
        cur = self.head
        pre = None
        while cur is not None:
            if cur.item == item:
                if not pre:
                    self.head = cur.next
                else:
                    pre.next = cur.next
                return True
            else:
                pre = cur
                cur = cur.next
    def change(self,item,index):
        if index > (self.length() - 1):
            self.append(item)
        else:
            cur = self.head
            for i in range(index - 1):
                cur = cur.next
            cur.item = item
    def print(self):
        cur = self.head
        while cur is not None:
            print(cur.item)
            cur = cur.next
l = LinkList()
for i in range(4):
    l.append(i)
l.print()
l.delete(3)
l.print()
print(l.find(1))