st = [False]*16
ans = []
st[0] = True

dt = {
    0:'人和狼过河了',
    1:'人和羊过河了',
    2:'人和菜过河了',
    3:'人自己过河了'
}
nxt = lambda a,b,c,d:a<<3|b<<2|c<<1|d
def dfs(cur):
    if cur == 15:
        print(' '.join(ans))
        return
    peo,wlof,sheep,cab = ((cur>>i)&1 for i in range(3,-1,-1))
    sta = (peo,wlof,sheep,cab,peo)
    if peo!= sheep and (sheep == cab or wlof == sheep): return
    num= (w,s,c,d) = nxt(1-peo,1-peo,sheep,cab),nxt(1-peo,wlof,1-sheep,cab),nxt(1-peo,wlof,sheep,1-cab),nxt(1-peo,wlof,sheep,cab)
    for i,j in enumerate(sta[1:]):
        if peo == j and (not st[num[i]]):
            st[num[i]] = True
            ans.append(dt[i])
            dfs(num[i])
            ans.pop()
            st[num[i]] = False
dfs(0)