FROM nginx:stable-alpine

LABEL maintainer="Iorimiya <iorimiya.vina@gmail.com>"
LABEL version="1.1.0"
LABEL description="Movie Box Office Predictor Website Node"

RUN apk add --no-cache libcap

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN addgroup -g $USER_GID $USERNAME \
    && adduser -u $USER_UID -G $USERNAME -D $USERNAME

RUN setcap 'cap_net_bind_service=+ep' /usr/sbin/nginx

RUN touch /var/run/nginx.pid \
    && chown -R $USERNAME:$USER_GID /var/cache/nginx /var/log/nginx /etc/nginx /usr/share/nginx/html /var/run/nginx.pid

RUN sed -i '1s/^/pid \/tmp\/nginx.pid;\n/' /etc/nginx/nginx.conf

RUN rm /etc/nginx/conf.d/default.conf

COPY --chown=$USERNAME:$USER_GID ./nginx/default.conf /etc/nginx/conf.d/default.conf

USER $USERNAME

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
