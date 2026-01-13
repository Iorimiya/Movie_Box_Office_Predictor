FROM nginx:stable-alpine

LABEL maintainer="Iorimiya <iorimiya.vina@gmail.com>"
LABEL version="1.1.0"
LABEL description="Movie Box Office Predictor Website Node"

# 安裝必要工具以授予特權連接埠權限
RUN apk add --no-cache libcap

# 建立與其他節點一致的非 root 使用者
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN addgroup -g $USER_GID $USERNAME \
    && adduser -u $USER_UID -G $USERNAME -D $USERNAME

RUN setcap 'cap_net_bind_service=+ep' /usr/sbin/nginx


RUN sed -i '/^user /d' /etc/nginx/nginx.conf \
    && sed -i '/^pid /d' /etc/nginx/nginx.conf \
    && sed -i '1s/^/pid \/tmp\/nginx.pid;\n/' /etc/nginx/nginx.conf


RUN touch /var/run/nginx.pid \
    && chown -R $USERNAME:$USER_GID /var/cache/nginx /var/log/nginx /etc/nginx /usr/share/nginx/html /var/run/nginx.pid

RUN rm /etc/nginx/conf.d/default.conf
COPY --chown=$USERNAME:$USER_GID ./nginx/default.conf /etc/nginx/conf.d/default.conf

# 切換身分
USER $USERNAME

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
